#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
from tf2_ros import TransformException
from geometry_msgs.msg import TransformStamped
from tf_transformations import quaternion_from_euler

from localization.icp_utils import (
    pose_to_T, T_to_pose, invert_T, icp_2d_point_to_point, wrap_angle
)


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def tf_to_T2(tf_msg) -> np.ndarray:
    q = tf_msg.transform.rotation
    yaw = math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )
    tx = tf_msg.transform.translation.x
    ty = tf_msg.transform.translation.y
    return pose_to_T(tx, ty, yaw)


class IcpScanToScan(Node):
    def __init__(self):
        super().__init__("icp_scan_to_scan")

        # Params
        self.declare_parameter("scan_topic", "/lidar/scan")
        self.declare_parameter("is_turning_topic", "/nav/is_turning")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("map_frame", "map")

        self.declare_parameter("downsample_step", 2)
        self.declare_parameter("scans_to_skip", 10)
        self.declare_parameter("range_min", 0.12)
        self.declare_parameter("range_max", 4.0)

        self.declare_parameter("icp_max_iter", 25)
        self.declare_parameter("icp_max_corr_dist", 0.35)
        self.declare_parameter("icp_min_inliers", 60)
        self.declare_parameter("icp_rmse_thresh", 0.20)
        self.declare_parameter("max_step_trans", 0.5)
        self.declare_parameter("max_step_rot_deg", 25.0)

        self.scan_topic = self.get_parameter("scan_topic").value
        self.scans_to_skip = self.get_parameter("scans_to_skip").value
        self.odom_frame = self.get_parameter("odom_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.map_frame = self.get_parameter("map_frame").value

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Subscribe
        self.create_subscription(LaserScan, self.scan_topic, self.cb_scan, 10)

        is_turning_topic = self.get_parameter("is_turning_topic").value
        self.is_turning_subscription = self.create_subscription(
            Bool,
            is_turning_topic,
            self.is_turning_callback,
            10)
        self.is_turning = False
        self.skipped_scans = 0

        # State
        self.prev_pts = None
        self.prev_stamp = None
        self.MTB = np.eye(3, dtype=np.float64)  # our internal "map pose" estimate

        self.get_logger().info("ICP scan-to-scan node started")

    def is_turning_callback(self, msg: Bool):
        self.is_turning = msg.data

    def scan_to_points(self, msg: LaserScan) -> np.ndarray:
        step = int(self.get_parameter("downsample_step").value)
        rmin = float(self.get_parameter("range_min").value)
        rmax = float(self.get_parameter("range_max").value)

        pts = []
        ang = msg.angle_min
        for i, r in enumerate(msg.ranges):
            if i % step != 0:
                ang += msg.angle_increment
                continue
            if not math.isfinite(r) or r < rmin or r > rmax:
                ang += msg.angle_increment
                continue
            pts.append([r * math.cos(ang), r * math.sin(ang)])
            ang += msg.angle_increment

        if len(pts) == 0:
            return np.zeros((0, 2), dtype=np.float64)
        return np.asarray(pts, dtype=np.float64)

    def lookup_OTB(self, stamp, timeout_sec=0.05) -> np.ndarray:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.odom_frame, self.base_frame, stamp
            )
            return tf_to_T2(tf)
        except TransformException as ex:
            self.get_logger().warn(f"TF lookup OTB failed: {ex}")
            return None

    def broadcast_map_odom(self, stamp, MTO: np.ndarray):
        tx, ty, yaw = T_to_pose(MTO)
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.odom_frame
        t.transform.translation.x = tx
        t.transform.translation.y = ty
        t.transform.translation.z = 0.0
        q = quaternion_from_euler(0.0, 0.0, yaw)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(t)

    def cb_scan(self, msg: LaserScan):
        if self.is_turning:
            return
        
        if self.skipped_scans < self.scans_to_skip:
            self.skipped_scans += 1
            return

        self.skipped_scans = 0
        pts = self.scan_to_points(msg)
        if pts.shape[0] < 60:
            return

        if self.prev_pts is None:
            self.prev_pts = pts
            self.prev_stamp = msg.header.stamp
            return

        # Initial guess from odometry: T_guess maps current base into previous base
        OTB_prev = self.lookup_OTB(self.prev_stamp)
        OTB_cur = self.lookup_OTB(msg.header.stamp)
        if OTB_prev is None or OTB_cur is None:
            self.prev_pts = pts
            self.prev_stamp = msg.header.stamp
            return

        # Relative motion in odom: prev->cur is inv(OTB_prev)*OTB_cur
        T_prev_to_cur = invert_T(OTB_prev) @ OTB_cur

        # For scan matching we want src(current) -> dst(prev):
        # That is approximately inv(prev->cur)
        T_guess = invert_T(T_prev_to_cur)

        # ICP: current scan (src) to previous scan (dst)
        T_icp, info = icp_2d_point_to_point(
            src_pts=pts,
            dst_pts=self.prev_pts,
            init_T=T_guess,
            max_iter=int(self.get_parameter("icp_max_iter").value),
            max_corr_dist=float(self.get_parameter("icp_max_corr_dist").value),
            min_inliers=int(self.get_parameter("icp_min_inliers").value),
        )

        # Quality gating
        rmse_thr = float(self.get_parameter("icp_rmse_thresh").value)
        if info["inliers"] < int(self.get_parameter("icp_min_inliers").value) or info["rmse"] > rmse_thr:
            self.get_logger().warn(f"ICP rejected: inliers={info['inliers']} rmse={info['rmse']:.3f}")
            self.prev_pts = pts
            self.prev_stamp = msg.header.stamp
            return

        # Extract the implied motion prev->cur:
        # ICP gives cur->prev (src->dst). So invert it.
        T_cur_to_prev = T_icp
        T_prev_to_cur_icp = invert_T(T_cur_to_prev)

        # Reject huge steps
        dx, dy, dth = T_to_pose(T_prev_to_cur_icp)
        max_trans = float(self.get_parameter("max_step_trans").value)
        max_rot = math.radians(float(self.get_parameter("max_step_rot_deg").value))
        if math.hypot(dx, dy) > max_trans or abs(wrap_angle(dth)) > max_rot:
            self.get_logger().warn(f"ICP step too large: d={math.hypot(dx,dy):.2f} th={math.degrees(dth):.1f}")
            self.prev_pts = pts
            self.prev_stamp = msg.header.stamp
            return

        # Integrate into our internal map pose estimate
        self.MTB = self.MTB @ T_prev_to_cur_icp

        # Publish map->odom
        # MTO = MTB * inv(OTB_cur)
        MTO = self.MTB @ invert_T(OTB_cur)
        self.broadcast_map_odom(msg.header.stamp, MTO)

        self.get_logger().info(f"ICP ok: inliers={info['inliers']} rmse={info['rmse']:.3f}")

        # Update previous scan
        self.prev_pts = pts
        self.prev_stamp = msg.header.stamp


def main():
    rclpy.init()
    node = IcpScanToScan()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()