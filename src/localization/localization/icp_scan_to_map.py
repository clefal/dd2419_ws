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
    pose_to_T, T_to_pose, invert_T, apply_T, icp_2d_point_to_point, wrap_angle
)


def tf_to_T2(tf_msg) -> np.ndarray:
    q = tf_msg.transform.rotation
    yaw = math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )
    tx = tf_msg.transform.translation.x
    ty = tf_msg.transform.translation.y
    return pose_to_T(tx, ty, yaw)


def voxel_downsample_2d(pts: np.ndarray, voxel: float) -> np.ndarray:
    """Simple grid hash downsample."""
    if pts.shape[0] == 0:
        return pts
    key = np.floor(pts / voxel).astype(np.int32)
    _, idx = np.unique(key, axis=0, return_index=True)
    return pts[idx]


class IcpScanToMap(Node):
    def __init__(self):
        super().__init__("icp_scan_to_map")

        # Params
        self.declare_parameter("scan_topic", "/lidar/scan")
        self.declare_parameter("is_turning_topic", "/nav/is_turning")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("map_frame", "map")

        self.declare_parameter("downsample_step", 2)
        self.declare_parameter("scans_to_skip", 5)
        self.declare_parameter("range_min", 0.1)
        self.declare_parameter("range_max", 4.0)

        self.declare_parameter("icp_max_iter", 30)
        self.declare_parameter("icp_max_corr_dist", 0.35)
        self.declare_parameter("icp_min_inliers", 80)
        self.declare_parameter("icp_rmse_thresh", 0.18)

        self.declare_parameter("keyframe_min_trans", 0.25)
        self.declare_parameter("keyframe_min_rot_deg", 12.0)
        self.declare_parameter("map_voxel", 0.05)
        self.declare_parameter("map_max_points", 60000)

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
        self.MTO = np.eye(3, dtype=np.float64)  # current correction
        self.map_pts = None                    # Nx2 points in map frame (our ICP target)
        self.last_key_MTB = None               # last keyframe robot pose in map

        self.get_logger().info("ICP scan-to-map node started")

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

    def lookup_OTB(self, stamp) -> np.ndarray:
        try:
            tf = self.tf_buffer.lookup_transform(self.odom_frame, self.base_frame, stamp, timeout=rclpy.time.Duration(seconds=1))
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

    def should_add_keyframe(self, MTB: np.ndarray) -> bool:
        if self.last_key_MTB is None:
            return True
        dT = invert_T(self.last_key_MTB) @ MTB
        dx, dy, dth = T_to_pose(dT)
        if math.hypot(dx, dy) >= float(self.get_parameter("keyframe_min_trans").value):
            return True
        if abs(wrap_angle(dth)) >= math.radians(float(self.get_parameter("keyframe_min_rot_deg").value)):
            return True
        return False

    def add_scan_to_map(self, scan_pts_lidar: np.ndarray, MTB: np.ndarray):
        pts_map = apply_T(MTB, scan_pts_lidar)
        pts_map = voxel_downsample_2d(pts_map, float(self.get_parameter("map_voxel").value))

        if self.map_pts is None:
            self.map_pts = pts_map
        else:
            self.map_pts = np.vstack([self.map_pts, pts_map])
            # limit size
            maxN = int(self.get_parameter("map_max_points").value)
            if self.map_pts.shape[0] > maxN:
                self.map_pts = self.map_pts[-maxN:, :]

        self.map_pts = voxel_downsample_2d(self.map_pts, float(self.get_parameter("map_voxel").value))

    def cb_scan(self, msg: LaserScan):
        if self.is_turning:
            return
        
        if self.skipped_scans < self.scans_to_skip:
            self.skipped_scans += 1
            return

        self.skipped_scans = 0
        self.get_logger().info("Processing scan")

        scan_pts = self.scan_to_points(msg)

        self.get_logger().info("Processing scan 2")
        if scan_pts.shape[0] < 80:
            return

        OTB = self.lookup_OTB(msg.header.stamp)

        self.get_logger().info("Processing scan 3")
        if OTB is None:
            return

        # Current predicted robot pose in map using current correction:
        # MTB_pred = MTO * OTB
        MTB_pred = self.MTO @ OTB

        self.get_logger().info("Processing scan 4")

        # Bootstrap map from first scan (like slides suggest) :contentReference[oaicite:1]{index=1}
        if self.map_pts is None:
            self.add_scan_to_map(scan_pts, MTB_pred)
            self.last_key_MTB = MTB_pred
            self.broadcast_map_odom(msg.header.stamp, self.MTO)
            self.get_logger().info("Initialized ICP map from first scan")
            return

        # ICP: align current scan to map_pts
        # We align in MAP frame, so source should be scan already roughly in map frame.
        src0 = apply_T(MTB_pred, scan_pts)  # initial placement
        # Run ICP from src0 -> map
        T_icp, info = icp_2d_point_to_point(
            src_pts=src0,
            dst_pts=self.map_pts,
            init_T=np.eye(3, dtype=np.float64),
            max_iter=int(self.get_parameter("icp_max_iter").value),
            max_corr_dist=float(self.get_parameter("icp_max_corr_dist").value),
            min_inliers=int(self.get_parameter("icp_min_inliers").value),
        )

        rmse_thr = float(self.get_parameter("icp_rmse_thresh").value)
        if info["inliers"] < int(self.get_parameter("icp_min_inliers").value) or info["rmse"] > rmse_thr:
            self.get_logger().warn(f"ICP rejected: inliers={info['inliers']} rmse={info['rmse']:.3f}")
            # still broadcast previous correction so TF exists
            self.broadcast_map_odom(msg.header.stamp, self.MTO)
            return

        # T_icp maps src0 -> map (i.e., refines placement in map frame)
        # So corrected MTB is:
        MTB_new = T_icp @ MTB_pred

        # Update correction MTO = MTB_new * inv(OTB)
        self.MTO = MTB_new @ invert_T(OTB)

        # Publish map->odom
        self.broadcast_map_odom(msg.header.stamp, self.MTO)
        self.get_logger().info(f"ICP ok: inliers={info['inliers']} rmse={info['rmse']:.3f}")

        # Add to map if moved enough (keyframe logic)
        if self.should_add_keyframe(MTB_new):
            self.add_scan_to_map(scan_pts, MTB_new)
            self.last_key_MTB = MTB_new


def main():
    rclpy.init()
    node = IcpScanToMap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()