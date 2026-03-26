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

def median_filter_ranges(ranges, kernel_size=5):
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    half = kernel_size // 2

    arr = np.array(ranges, dtype=float)
    out = arr.copy()

    n = len(arr)

    for i in range(n):
        window_vals = []

        for j in range(max(0, i - half), min(n, i + half + 1)):
            v = arr[j]
            if math.isfinite(v):
                window_vals.append(v)

        if len(window_vals) > 0:
            out[i] = float(np.median(window_vals))
        else:
            out[i] = np.nan

    return out

def remove_isolated_points_ordered(pts, neighbor_dist_thresh=0.12):
    """
    pts: array Nx2 ordenado como sale del scan
    conserva puntos que tengan al menos un vecino cercano
    """
    if pts.shape[0] < 3:
        return pts

    keep = np.zeros(pts.shape[0], dtype=bool)

    keep[0] = np.linalg.norm(pts[1] - pts[0]) < neighbor_dist_thresh
    keep[-1] = np.linalg.norm(pts[-1] - pts[-2]) < neighbor_dist_thresh

    for i in range(1, pts.shape[0] - 1):
        d_prev = np.linalg.norm(pts[i] - pts[i - 1])
        d_next = np.linalg.norm(pts[i] - pts[i + 1])

        if d_prev < neighbor_dist_thresh or d_next < neighbor_dist_thresh:
            keep[i] = True

    return pts[keep]

class IcpScanToScan(Node):
    def __init__(self):
        super().__init__("icp_scan_to_scan")

        # Params
        self.declare_parameter("scan_topic", "/lidar/scan")
        self.declare_parameter("is_turning_topic", "/nav/is_turning")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("map_frame", "map")

        self.declare_parameter("downsample_step", 1)
        self.declare_parameter("scans_to_skip", 1)
        self.declare_parameter("range_min", 0.05)
        self.declare_parameter("range_max", 50.0)

        self.declare_parameter("icp_max_iter", 60) # Max number of iterations
        self.declare_parameter("icp_tol", 1e-6) # Convergence tolerance for RMSE
        self.declare_parameter("icp_max_corr_dist", 0.10) # Max distance between corresponding points - Defines what points are matched (modify this one for better results)
        self.declare_parameter("icp_min_inliers", 60) # Min number of inliers
        self.declare_parameter("icp_rmse_thresh", 0.15) # RMSE threshold

        self.declare_parameter("max_step_trans", 0.5) # Max step size in meters
        self.declare_parameter("max_step_rot_deg", 25.0) # Max step size in degrees

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
        self.create_subscription(
            Bool,
            is_turning_topic,
            self.is_turning_callback,
            10)
        self.is_turning = False
        self.skipped_scans = 0

        # State
        self.prev_pts = None
        self.prev_stamp = None
        # self.MTB = np.eye(3, dtype=np.float64)  # our internal "map pose" estimate
        self.try_initialize_mtb()

        self.get_logger().info("ICP scan-to-scan node started")

    def is_turning_callback(self, msg: Bool):
        self.is_turning = msg.data

    def try_initialize_mtb(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame,
                "start",
                rclpy.time.Time(seconds=0),
                timeout=rclpy.time.Duration(seconds=1),
            )
            self.MTB = tf_to_T2(tf)
            self.get_logger().info("Initialized MTB from TF: map -> start")
        except TransformException as ex:
            self.MTB = np.eye(3, dtype=np.float64)
            self.get_logger().warn(
                f"Could not initialize from map->base_link TF, using identity: {ex}"
            )
            return False

    def scan_to_points(self, msg: LaserScan) -> np.ndarray:
        step = int(self.get_parameter("downsample_step").value)
        rmin = float(self.get_parameter("range_min").value)
        rmax = float(self.get_parameter("range_max").value)

        pts = []
        ang = msg.angle_min
        self.get_logger().info(f"Scan has {len(msg.ranges)} ranges")
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

    def lookup_OTB(self, stamp, timeout_sec=1) -> np.ndarray:
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
        # if self.is_turning:
        #     return
        
        if self.skipped_scans < self.scans_to_skip:
            self.skipped_scans += 1
            return
        
        self.skipped_scans = 0

        self.get_logger().info("New scan to process")
        
        # Filter scan
        msg.ranges = median_filter_ranges(msg.ranges, kernel_size=11)

        # Remove isolated points
        pts = remove_isolated_points_ordered(self.scan_to_points(msg), 0.12)

        self.get_logger().info(f"Valid scan has {pts.shape[0]} points")
        if pts.shape[0] < 60:
            self.get_logger().info("Valid scan too small")
            return

        if self.prev_pts is None:
            self.prev_pts = pts
            self.prev_stamp = msg.header.stamp
            return

        # Initial guess from odometry: T_prev_to_cur_guess maps current base into previous base
        OTB_prev = self.lookup_OTB(self.prev_stamp)
        OTB_cur = self.lookup_OTB(msg.header.stamp)
        if OTB_prev is None or OTB_cur is None:
            self.prev_pts = pts
            self.prev_stamp = msg.header.stamp
            return

        # This maps current base into previous base
        T_cur_in_prev_guess = invert_T(OTB_prev) @ OTB_cur

        # For scan matching we want src(current) -> dst(prev):
        # ICP: current scan (src) to previous scan (dst)
        T_icp, info = icp_2d_point_to_point(
            src_pts=pts,
            dst_pts=self.prev_pts,
            init_T=T_cur_in_prev_guess,
            max_iter=int(self.get_parameter("icp_max_iter").value),
            max_corr_dist=float(self.get_parameter("icp_max_corr_dist").value),
            min_inliers=int(self.get_parameter("icp_min_inliers").value),
            tol=float(self.get_parameter("icp_tol").value),
        )

        # Quality gating
        rmse_thr = float(self.get_parameter("icp_rmse_thresh").value)
        if info["inliers"] < int(self.get_parameter("icp_min_inliers").value) or info["rmse"] > rmse_thr:
            self.get_logger().warn(f"ICP rejected: inliers={info['inliers']} rmse={info['rmse']:.3f}")
            self.prev_pts = pts
            self.prev_stamp = msg.header.stamp
            return

        # Reject huge steps
        dx, dy, dth = T_to_pose(T_icp)
        max_trans = float(self.get_parameter("max_step_trans").value)
        max_rot = math.radians(float(self.get_parameter("max_step_rot_deg").value))
        if math.hypot(dx, dy) > max_trans or abs(wrap_angle(dth)) > max_rot:
            self.get_logger().warn(f"ICP step too large: d={math.hypot(dx,dy):.2f} th={math.degrees(dth):.1f}")
            self.prev_pts = pts
            self.prev_stamp = msg.header.stamp
            return

        # Integrate into our internal map pose estimate
        self.MTB = self.MTB @ T_icp

        # Publish map->odom
        # MTO = MTB * inv(OTB_cur)
        MTO = self.MTB @ invert_T(OTB_cur)
        self.broadcast_map_odom(msg.header.stamp, MTO)

        self.get_logger().info(f"ICP ok: inliers={info['inliers']} rmse={info['rmse']:.3f} iterations={info['iterations']}")

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