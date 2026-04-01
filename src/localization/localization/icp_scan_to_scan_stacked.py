#!/usr/bin/env python3
import math
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String
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


def transform_points(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    if pts is None or pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    R = T[:2, :2]
    t = T[:2, 2]
    return (pts @ R.T) + t


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


def reject_range_spikes(ranges, jump_thresh=0.25):
    if jump_thresh <= 0.0:
        return np.array(ranges, dtype=float)

    arr = np.array(ranges, dtype=float)
    out = arr.copy()

    if arr.shape[0] < 3:
        return out

    for i in range(1, arr.shape[0] - 1):
        cur = arr[i]
        prev = arr[i - 1]
        nxt = arr[i + 1]

        if not math.isfinite(cur):
            continue

        prev_ok = math.isfinite(prev)
        next_ok = math.isfinite(nxt)

        if prev_ok and next_ok:
            if abs(cur - prev) > jump_thresh and abs(cur - nxt) > jump_thresh:
                out[i] = np.nan

    return out


def remove_small_clusters_ordered(pts: np.ndarray, cluster_dist_thresh=0.12, min_cluster_size=3) -> np.ndarray:
    if pts.shape[0] == 0 or min_cluster_size <= 1:
        return pts

    clusters = []
    start = 0
    for i in range(1, pts.shape[0]):
        if np.linalg.norm(pts[i] - pts[i - 1]) > cluster_dist_thresh:
            if i - start >= min_cluster_size:
                clusters.append(pts[start:i])
            start = i

    if pts.shape[0] - start >= min_cluster_size:
        clusters.append(pts[start:])

    if not clusters:
        return np.zeros((0, 2), dtype=np.float64)

    return np.vstack(clusters)


class IcpScanToScan(Node):
    def __init__(self):
        super().__init__("icp_scan_to_scan")

        # Topics / frames
        self.declare_parameter("scan_topic", "/lidar/scan")
        self.declare_parameter("is_turning_topic", "/nav/is_turning")
        self.declare_parameter("debug_topic", "/localization/icp_debug")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("start_frame", "start")

        # Pre-processing
        self.declare_parameter("downsample_step", 1)
        self.declare_parameter("range_min", 0.05)
        self.declare_parameter("range_max", 15.0)
        self.declare_parameter("median_kernel_size", 11)
        self.declare_parameter("range_jump_thresh", 0.20)
        self.declare_parameter("neighbor_dist_thresh", 0.10)
        self.declare_parameter("cluster_dist_thresh", 0.10)
        self.declare_parameter("min_cluster_size", 4)
        self.declare_parameter("min_scan_points", 60)

        # Stacking
        self.declare_parameter("stack_size", 2)
        self.declare_parameter("stack_voxel_size", 0.06)  # 0 disables voxel reduction
        self.declare_parameter("stack_min_keyframe_translation", 0.05)
        self.declare_parameter("stack_min_keyframe_rotation_deg", 2.0)

        # ICP
        self.declare_parameter("icp_max_iter", 60)
        self.declare_parameter("icp_tol", 1e-6)
        self.declare_parameter("icp_max_corr_dist", 0.08)
        self.declare_parameter("icp_min_inliers", 80)
        self.declare_parameter("icp_rmse_thresh", 0.10)

        # Safety gates
        self.declare_parameter("max_step_trans", 0.5)
        self.declare_parameter("max_step_rot_deg", 25.0)
        self.declare_parameter("max_correction_trans", 0.12)
        self.declare_parameter("max_correction_rot_deg", 8.0)

        # Debug / runtime info
        self.declare_parameter("verbose", True)
        self.declare_parameter("publish_debug_topic", True)
        self.declare_parameter("log_rejections", True)

        self.scan_topic = self.get_parameter("scan_topic").value
        self.is_turning_topic = self.get_parameter("is_turning_topic").value
        self.debug_topic = self.get_parameter("debug_topic").value
        self.odom_frame = self.get_parameter("odom_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.map_frame = self.get_parameter("map_frame").value
        self.start_frame = self.get_parameter("start_frame").value

        self.verbose = bool(self.get_parameter("verbose").value)
        self.publish_debug_topic = bool(self.get_parameter("publish_debug_topic").value)
        self.log_rejections = bool(self.get_parameter("log_rejections").value)

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Subscriptions
        self.create_subscription(LaserScan, self.scan_topic, self.cb_scan, 10)
        self.create_subscription(Bool, self.is_turning_topic, self.is_turning_callback, 10)

        # Optional debug publisher
        self.debug_pub = None
        if self.publish_debug_topic:
            self.debug_pub = self.create_publisher(String, self.debug_topic, 10)

        # State
        self.scan_history = deque(maxlen=max(1, int(self.get_parameter("stack_size").value)))
        self.is_turning = False
        self.MTB = np.eye(3, dtype=np.float64)
        self.try_initialize_mtb()

        self.last_debug_msg = ""
        self.get_logger().info("ICP scan-to-scan node started")

    def is_turning_callback(self, msg: Bool):
        was_turning = self.is_turning
        self.is_turning = msg.data

        if was_turning and not self.is_turning:
            self.scan_history.clear()
            self.log_icp_status(
                "info",
                "Turning ended: cleared stacked scan history and waiting for fresh scans"
            )

    def try_initialize_mtb(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.start_frame,
                rclpy.time.Time(seconds=0),
                timeout=rclpy.time.Duration(seconds=5),
            )
            self.MTB = tf_to_T2(tf)
            self.get_logger().info(
                f"Initialized MTB from TF: {self.map_frame} -> {self.start_frame}"
            )
            return True
        except TransformException as ex:
            self.MTB = np.eye(3, dtype=np.float64)
            self.get_logger().warn(
                f"Could not initialize from {self.map_frame}->{self.start_frame} TF, using identity: {ex}"
            )
            return False

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

    def voxel_downsample_2d(self, pts: np.ndarray, voxel_size: float) -> np.ndarray:
        if pts.shape[0] == 0 or voxel_size <= 0.0:
            return pts

        ij = np.floor(pts / voxel_size).astype(np.int64)
        _, unique_idx = np.unique(ij, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)
        return pts[unique_idx]

    def lookup_OTB(self, stamp) -> np.ndarray:
        try:
            self.get_logger().info(f"Trying to lookup TF {self.odom_frame}->{self.base_frame}")
            tf = self.tf_buffer.lookup_transform(
                self.odom_frame, self.base_frame, stamp
            )
            return tf_to_T2(tf)
        except TransformException as ex:
            self.get_logger().warn(f"TF lookup {self.odom_frame}->{self.base_frame} failed: {ex}")
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

    def publish_debug(self, text: str):
        self.last_debug_msg = text
        if self.debug_pub is not None:
            msg = String()
            msg.data = text
            self.debug_pub.publish(msg)

    def log_icp_status(self, level: str, text: str):
        self.publish_debug(text)
        if level == "warn":
            self.get_logger().warn(text)
        else:
            self.get_logger().info(text)

    def should_add_keyframe(self, OTB_cur: np.ndarray) -> bool:
        if len(self.scan_history) == 0:
            return True

        last_OTB = self.scan_history[-1]["OTB"]
        T_cur_in_last = invert_T(last_OTB) @ OTB_cur
        dx, dy, dth = T_to_pose(T_cur_in_last)

        min_trans = float(self.get_parameter("stack_min_keyframe_translation").value)
        min_rot = math.radians(float(self.get_parameter("stack_min_keyframe_rotation_deg").value))

        return (math.hypot(dx, dy) >= min_trans) or (abs(wrap_angle(dth)) >= min_rot)

    def add_scan_to_history(self, pts: np.ndarray, stamp, OTB: np.ndarray, MTB: np.ndarray):
        if not self.should_add_keyframe(OTB):
            return

        self.scan_history.append({
            "pts": pts,
            "stamp": stamp,
            "OTB": OTB,
            "MTB": MTB.copy(),
        })

    def build_stacked_target(self):
        if len(self.scan_history) == 0:
            return None, None, None

        ref = self.scan_history[-1]
        OTB_ref = ref["OTB"]
        MTB_ref = ref["MTB"]

        stacked = []
        for item in self.scan_history:
            T_item_to_ref = invert_T(OTB_ref) @ item["OTB"]
            pts_ref = transform_points(item["pts"], T_item_to_ref)
            stacked.append(pts_ref)

        dst_pts = np.vstack(stacked) if len(stacked) > 0 else np.zeros((0, 2), dtype=np.float64)

        voxel = float(self.get_parameter("stack_voxel_size").value)
        dst_pts = self.voxel_downsample_2d(dst_pts, voxel)

        return dst_pts, ref["stamp"], OTB_ref, MTB_ref

    def preprocess_scan(self, msg: LaserScan) -> np.ndarray:
        kernel_size = int(self.get_parameter("median_kernel_size").value)
        kernel_size = max(1, kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1

        filtered_ranges = median_filter_ranges(msg.ranges, kernel_size=kernel_size)
        filtered_ranges = reject_range_spikes(
            filtered_ranges,
            jump_thresh=float(self.get_parameter("range_jump_thresh").value),
        )
        msg.ranges = filtered_ranges.tolist()

        pts = self.scan_to_points(msg)
        pts = remove_isolated_points_ordered(
            pts,
            float(self.get_parameter("neighbor_dist_thresh").value)
        )
        pts = remove_small_clusters_ordered(
            pts,
            cluster_dist_thresh=float(self.get_parameter("cluster_dist_thresh").value),
            min_cluster_size=int(self.get_parameter("min_cluster_size").value),
        )
        return pts

    def cb_scan(self, msg: LaserScan):
        if self.is_turning:
            return

        t0 = time.perf_counter()

        OTB_cur = self.lookup_OTB(msg.header.stamp)
        if OTB_cur is None:
            return

        pts = self.preprocess_scan(msg)

        min_scan_points = int(self.get_parameter("min_scan_points").value)
        if pts.shape[0] < min_scan_points:
            if self.log_rejections:
                self.log_icp_status(
                    "warn",
                    f"ICP skipped: too few points ({pts.shape[0]} < {min_scan_points})"
                )
            return

        if len(self.scan_history) == 0:
            self.add_scan_to_history(pts, msg.header.stamp, OTB_cur, self.MTB)
            self.log_icp_status("info", f"ICP initialized with first scan ({pts.shape[0]} points)")
            return

        dst_pts, ref_stamp, OTB_ref, MTB_ref = self.build_stacked_target()
        if dst_pts is None or dst_pts.shape[0] < min_scan_points:
            self.add_scan_to_history(pts, msg.header.stamp, OTB_cur, self.MTB)
            if self.log_rejections:
                self.log_icp_status("warn", "ICP skipped: stacked target not large enough")
            return

        # Current base into reference base (reference is last scan in stack)
        T_cur_in_ref_guess = invert_T(OTB_ref) @ OTB_cur
        guess_dx, guess_dy, guess_dth = T_to_pose(T_cur_in_ref_guess)

        T_icp, info = icp_2d_point_to_point(
            src_pts=pts,
            dst_pts=dst_pts,
            init_T=T_cur_in_ref_guess,
            max_iter=int(self.get_parameter("icp_max_iter").value),
            max_corr_dist=float(self.get_parameter("icp_max_corr_dist").value),
            min_inliers=int(self.get_parameter("icp_min_inliers").value),
            tol=float(self.get_parameter("icp_tol").value),
        )

        rmse_thr = float(self.get_parameter("icp_rmse_thresh").value)
        min_inliers = int(self.get_parameter("icp_min_inliers").value)

        if info["inliers"] < min_inliers or info["rmse"] > rmse_thr:
            if self.log_rejections:
                self.log_icp_status(
                    "warn",
                    "ICP rejected: "
                    f"inliers={info['inliers']}<{min_inliers} "
                    f"rmse={info['rmse']:.4f}>{rmse_thr:.4f} "
                    f"stack_scans={len(self.scan_history)} dst_pts={dst_pts.shape[0]}"
                )
            return

        dx, dy, dth = T_to_pose(T_icp)
        max_trans = float(self.get_parameter("max_step_trans").value)
        max_rot = math.radians(float(self.get_parameter("max_step_rot_deg").value))
        T_correction = T_icp @ invert_T(T_cur_in_ref_guess)
        corr_dx, corr_dy, corr_dth = T_to_pose(T_correction)
        max_corr_trans = float(self.get_parameter("max_correction_trans").value)
        max_corr_rot = math.radians(float(self.get_parameter("max_correction_rot_deg").value))

        if math.hypot(corr_dx, corr_dy) > max_corr_trans or abs(wrap_angle(corr_dth)) > max_corr_rot:
            if self.log_rejections:
                self.log_icp_status(
                    "warn",
                    "ICP rejected: correction too large "
                    f"(d={math.hypot(corr_dx, corr_dy):.3f} m, "
                    f"yaw={math.degrees(corr_dth):.2f} deg)"
                )
            return

        if math.hypot(dx, dy) > max_trans or abs(wrap_angle(dth)) > max_rot:
            if self.log_rejections:
                self.log_icp_status(
                    "warn",
                    "ICP rejected: step too large "
                    f"(d={math.hypot(dx, dy):.3f} m, yaw={math.degrees(dth):.2f} deg)"
                )
            return

        # Integrate estimate
        self.MTB = MTB_ref @ T_icp

        # map->odom = map->base * inv(odom->base)
        MTO = self.MTB @ invert_T(OTB_cur)
        self.broadcast_map_odom(msg.header.stamp, MTO)

        dt_ms = (time.perf_counter() - t0) * 1000.0
        status = (
            "ICP ok: "
            f"src_pts={pts.shape[0]} "
            f"dst_pts={dst_pts.shape[0]} "
            f"stack_scans={len(self.scan_history)} "
            f"guess=({guess_dx:+.3f}, {guess_dy:+.3f}, {math.degrees(guess_dth):+.2f}deg) "
            f"result=({dx:+.3f}, {dy:+.3f}, {math.degrees(dth):+.2f}deg) "
            f"corr=({corr_dx:+.3f}, {corr_dy:+.3f}, {math.degrees(corr_dth):+.2f}deg) "
            f"inliers={info['inliers']} "
            f"rmse={info['rmse']:.4f} "
            f"iters={info['iterations']} "
            f"dt={dt_ms:.1f}ms "
        )
        self.log_icp_status("info", status)

        self.add_scan_to_history(pts, msg.header.stamp, OTB_cur, self.MTB)


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
