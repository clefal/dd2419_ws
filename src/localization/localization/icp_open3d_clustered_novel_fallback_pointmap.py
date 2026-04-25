#!/usr/bin/env python3
"""
Standalone ROS 2 node for 2D LaserScan scan-to-map point-cloud ICP using Open3D, with significant-cluster filtering and optional novel-point map insertion.

What it does
------------
- Subscribes to a LaserScan topic.
- Uses odom->base and base->laser TF as the motion prior.
- Accumulates scans until enough motion/time has passed.
- Builds a point-cloud map in the map frame.
- Filters scan points into significant clusters before ICP/map insertion.
- Optionally inserts only points that are novel with respect to the current map,
  avoiding repeated insertion of already-mapped walls.
- Runs Open3D ICP from the accumulated filtered scan to the point-cloud map.
- If the ICP result is accepted, updates and broadcasts map->odom.
- Publishes RViz-friendly PointCloud2 topics:
    * /localization/open3d_point_map
    * /localization/open3d_stacked_scan
    * /localization/open3d_aligned_scan

This file intentionally does NOT import any of the previous ICP implementation files.
"""

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener


# =============================================================================
# Small 2D transform helpers
# =============================================================================


def wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def pose_to_matrix(x: float, y: float, theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array(
        [[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def matrix_to_pose(T: np.ndarray) -> Tuple[float, float, float]:
    return float(T[0, 2]), float(T[1, 2]), math.atan2(float(T[1, 0]), float(T[0, 0]))


def invert_transform(T: np.ndarray) -> np.ndarray:
    R = T[:2, :2]
    t = T[:2, 2]
    Tinv = np.eye(3, dtype=float)
    Tinv[:2, :2] = R.T
    Tinv[:2, 2] = -R.T @ t
    return Tinv


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if pts.shape[0] == 0:
        return pts.copy()
    return pts @ T[:2, :2].T + T[:2, 2]


def relative_pose(T_from: np.ndarray, T_to: np.ndarray) -> Tuple[float, float, float]:
    return matrix_to_pose(invert_transform(T_from) @ T_to)


def interpolate_pose(T_old: np.ndarray, T_new: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    x0, y0, th0 = matrix_to_pose(T_old)
    x1, y1, th1 = matrix_to_pose(T_new)
    x = (1.0 - alpha) * x0 + alpha * x1
    y = (1.0 - alpha) * y0 + alpha * y1
    th = wrap_angle(th0 + alpha * wrap_angle(th1 - th0))
    return pose_to_matrix(x, y, th)


def tfmsg_to_matrix(tf_msg: TransformStamped) -> np.ndarray:
    t = tf_msg.transform.translation
    q = tf_msg.transform.rotation
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return pose_to_matrix(t.x, t.y, yaw)


def matrix_to_tf_msg(T: np.ndarray, parent: str, child: str, stamp) -> TransformStamped:
    x, y, yaw = matrix_to_pose(T)
    msg = TransformStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = parent
    msg.child_frame_id = child
    msg.transform.translation.x = float(x)
    msg.transform.translation.y = float(y)
    msg.transform.translation.z = 0.0
    cy = math.cos(0.5 * yaw)
    sy = math.sin(0.5 * yaw)
    msg.transform.rotation.x = 0.0
    msg.transform.rotation.y = 0.0
    msg.transform.rotation.z = sy
    msg.transform.rotation.w = cy
    return msg


def matrix_2d_to_4d(T2: np.ndarray) -> np.ndarray:
    T4 = np.eye(4, dtype=float)
    T4[0, 0] = T2[0, 0]
    T4[0, 1] = T2[0, 1]
    T4[1, 0] = T2[1, 0]
    T4[1, 1] = T2[1, 1]
    T4[0, 3] = T2[0, 2]
    T4[1, 3] = T2[1, 2]
    return T4


def matrix_4d_to_2d(T4: np.ndarray) -> np.ndarray:
    yaw = math.atan2(float(T4[1, 0]), float(T4[0, 0]))
    return pose_to_matrix(float(T4[0, 3]), float(T4[1, 3]), yaw)


# =============================================================================
# Point cloud utilities
# =============================================================================


def scan_to_points(
    scan: LaserScan,
    range_min_clip: float,
    range_max_clip: Optional[float],
    stride: int,
) -> np.ndarray:
    ranges = np.asarray(scan.ranges, dtype=float)
    if ranges.size == 0:
        return np.zeros((0, 2), dtype=float)

    rmax = scan.range_max if range_max_clip is None else min(float(scan.range_max), float(range_max_clip))
    valid = np.isfinite(ranges)
    valid &= ranges >= max(float(scan.range_min), float(range_min_clip))
    valid &= ranges <= rmax

    stride = max(1, int(stride))
    if stride > 1:
        stride_mask = np.zeros(ranges.shape[0], dtype=bool)
        stride_mask[::stride] = True
        valid &= stride_mask

    idx = np.nonzero(valid)[0]
    if idx.size == 0:
        return np.zeros((0, 2), dtype=float)

    rr = ranges[idx]
    angles = float(scan.angle_min) + idx.astype(float) * float(scan.angle_increment)
    pts = np.empty((idx.size, 2), dtype=float)
    pts[:, 0] = rr * np.cos(angles)
    pts[:, 1] = rr * np.sin(angles)
    return pts


def points2d_to_o3d(points: np.ndarray) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    if points.shape[0] == 0:
        cloud.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=float))
        return cloud
    xyz = np.zeros((points.shape[0], 3), dtype=float)
    xyz[:, :2] = points[:, :2]
    cloud.points = o3d.utility.Vector3dVector(xyz)
    return cloud


def voxel_downsample_2d(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if points.shape[0] == 0 or voxel_size <= 0.0:
        return points.copy()

    # Fast deterministic 2D voxel filter: keep one point per grid cell.
    keys = np.floor(points[:, :2] / float(voxel_size)).astype(np.int64)
    _, keep = np.unique(keys, axis=0, return_index=True)
    keep = np.sort(keep)
    return points[keep]


def limit_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    # Keep recent-ish/random coverage without deterministic spatial bias.
    idx = np.linspace(0, points.shape[0] - 1, max_points).astype(np.int64)
    return points[idx]


def points_to_pointcloud2(points: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    msg = PointCloud2()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = int(points.shape[0])
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True

    if points.shape[0] == 0:
        msg.data = b""
        return msg

    xyz = np.zeros((points.shape[0], 3), dtype=np.float32)
    xyz[:, :2] = points.astype(np.float32, copy=False)
    msg.data = xyz.tobytes()
    return msg


@dataclass
class BufferedScan:
    points_laser: np.ndarray
    T_odom_laser: np.ndarray


@dataclass
class IcpOutcome:
    T_map_laser: np.ndarray
    fitness: float
    rmse: float
    correspondences: int
    elapsed_ms: float
    converged: bool


class Open3DClusteredPointMapLocalization(Node):
    def __init__(self) -> None:
        super().__init__("open3d_clustered_novel_fallback_pointmap_localization")

        # Topics and frames
        self.declare_parameter("scan_topic", "/localization/preprocessed_scan")
        self.declare_parameter("is_turning_topic", "/nav/is_turning")
        self.declare_parameter("base_frame", "base_link_temp")
        self.declare_parameter("odom_frame", "odom_temp")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("laser_base_frame", "base_link")
        self.declare_parameter("initial_map_to_start_frame", "start")

        # RViz topics
        self.declare_parameter("point_map_topic", "/localization/open3d_point_map")
        self.declare_parameter("stacked_scan_topic", "/localization/open3d_stacked_scan")
        self.declare_parameter("aligned_scan_topic", "/localization/open3d_aligned_scan")

        # Scan preprocessing
        self.declare_parameter("range_min_clip", 0.15)
        self.declare_parameter("range_max_clip", 5.0)
        self.declare_parameter("scan_stride", 1)
        self.declare_parameter("min_scan_points", 80)

        # Cluster filtering
        self.declare_parameter("enable_cluster_filter", True)
        self.declare_parameter("cluster_jump_thresh_near", 0.16)
        self.declare_parameter("cluster_jump_thresh_far", 0.55)
        self.declare_parameter("cluster_min_points", 8)
        self.declare_parameter("cluster_min_extent", 0.40)
        self.declare_parameter("cluster_max_mean_range", 5.0)

        # Motion-trigger accumulation
        self.declare_parameter("trigger_translation", 0.20)
        self.declare_parameter("trigger_rotation_deg", 5.0)
        self.declare_parameter("trigger_max_scans", 14)
        self.declare_parameter("publish_wait_debug", True)

        # Open3D ICP — more flexible
        self.declare_parameter("open3d_max_corr_dist", 0.30)
        self.declare_parameter("open3d_max_iters", 70)
        self.declare_parameter("open3d_source_voxel", 0.035)
        self.declare_parameter("open3d_target_voxel", 0.08)
        self.declare_parameter("open3d_min_fitness", 0.30)
        self.declare_parameter("open3d_max_rmse", 0.13)
        self.declare_parameter("icp_accept_max_translation", 0.38)
        self.declare_parameter("icp_accept_max_rotation_deg", 8.0)
        self.declare_parameter("pose_smoothing_alpha", 0.06)

        # Point map maintenance
        self.declare_parameter("pointmap_voxel", 0.04)
        self.declare_parameter("pointmap_max_points", 40000)
        self.declare_parameter("pointmap_min_points", 200)

        # ICP-corrected map update gate — still stricter than localization
        self.declare_parameter("map_update_min_translation", 0.30)
        self.declare_parameter("map_update_min_rotation_deg", 10.0)
        self.declare_parameter("map_update_min_fitness", 0.60)
        self.declare_parameter("map_update_max_rmse", 0.085)
        self.declare_parameter("map_update_max_correction_translation", 0.12)
        self.declare_parameter("map_update_max_correction_rotation_deg", 3.5)

        # Novel-point insertion
        self.declare_parameter("insert_only_novel_points", True)
        self.declare_parameter("novel_point_min_dist", 0.12)
        self.declare_parameter("novel_point_max_dist", 5.0)
        self.declare_parameter("novel_min_points", 8)

        # Novel fallback
        self.declare_parameter("allow_novel_update_when_map_update_rejected", True)
        self.declare_parameter("novel_update_min_fitness", 0.45)
        self.declare_parameter("novel_update_max_rmse", 0.11)

        # Odom-only map update fallback
        self.declare_parameter("allow_odom_map_update_when_icp_rejected", True)
        self.declare_parameter("odom_map_update_min_translation", 0.30)
        self.declare_parameter("odom_map_update_min_rotation_deg", 10.0)
        self.declare_parameter("odom_map_update_max_fitness", 0.08)
        self.declare_parameter("odom_map_update_min_corr", 0)

        # Debug
        self.declare_parameter("log_icp_debug", True)

        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.is_turning_topic = str(self.get_parameter("is_turning_topic").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.map_frame = str(self.get_parameter("map_frame").value)
        self.laser_base_frame = str(self.get_parameter("laser_base_frame").value)
        self.initial_map_to_start_frame = str(self.get_parameter("initial_map_to_start_frame").value)

        self.point_map_topic = str(self.get_parameter("point_map_topic").value)
        self.stacked_scan_topic = str(self.get_parameter("stacked_scan_topic").value)
        self.aligned_scan_topic = str(self.get_parameter("aligned_scan_topic").value)

        self.range_min_clip = float(self.get_parameter("range_min_clip").value)
        self.range_max_clip = float(self.get_parameter("range_max_clip").value)
        self.scan_stride = int(self.get_parameter("scan_stride").value)
        self.min_scan_points = int(self.get_parameter("min_scan_points").value)

        self.enable_cluster_filter = bool(self.get_parameter("enable_cluster_filter").value)
        self.cluster_jump_thresh_near = float(self.get_parameter("cluster_jump_thresh_near").value)
        self.cluster_jump_thresh_far = float(self.get_parameter("cluster_jump_thresh_far").value)
        self.cluster_min_points = int(self.get_parameter("cluster_min_points").value)
        self.cluster_min_extent = float(self.get_parameter("cluster_min_extent").value)
        self.cluster_max_mean_range = float(self.get_parameter("cluster_max_mean_range").value)

        self.trigger_translation = float(self.get_parameter("trigger_translation").value)
        self.trigger_rotation_deg = float(self.get_parameter("trigger_rotation_deg").value)
        self.trigger_max_scans = max(1, int(self.get_parameter("trigger_max_scans").value))
        self.publish_wait_debug = bool(self.get_parameter("publish_wait_debug").value)

        self.open3d_max_corr_dist = float(self.get_parameter("open3d_max_corr_dist").value)
        self.open3d_max_iters = int(self.get_parameter("open3d_max_iters").value)
        self.open3d_source_voxel = float(self.get_parameter("open3d_source_voxel").value)
        self.open3d_target_voxel = float(self.get_parameter("open3d_target_voxel").value)
        self.open3d_min_fitness = float(self.get_parameter("open3d_min_fitness").value)
        self.open3d_max_rmse = float(self.get_parameter("open3d_max_rmse").value)
        self.icp_accept_max_translation = float(self.get_parameter("icp_accept_max_translation").value)
        self.icp_accept_max_rotation_deg = float(self.get_parameter("icp_accept_max_rotation_deg").value)
        self.pose_smoothing_alpha = float(self.get_parameter("pose_smoothing_alpha").value)

        self.pointmap_voxel = float(self.get_parameter("pointmap_voxel").value)
        self.pointmap_max_points = int(self.get_parameter("pointmap_max_points").value)
        self.pointmap_min_points = int(self.get_parameter("pointmap_min_points").value)
        self.insert_only_novel_points = bool(self.get_parameter("insert_only_novel_points").value)
        self.novel_point_min_dist = float(self.get_parameter("novel_point_min_dist").value)
        self.novel_point_max_dist = float(self.get_parameter("novel_point_max_dist").value)
        self.novel_min_points = int(self.get_parameter("novel_min_points").value)
        self.allow_novel_update_when_map_update_rejected = bool(
            self.get_parameter("allow_novel_update_when_map_update_rejected").value
        )
        self.novel_update_min_fitness = float(self.get_parameter("novel_update_min_fitness").value)
        self.novel_update_max_rmse = float(self.get_parameter("novel_update_max_rmse").value)
        self.map_update_min_translation = float(self.get_parameter("map_update_min_translation").value)
        self.map_update_min_rotation_deg = float(self.get_parameter("map_update_min_rotation_deg").value)
        self.map_update_min_fitness = float(self.get_parameter("map_update_min_fitness").value)
        self.map_update_max_rmse = float(self.get_parameter("map_update_max_rmse").value)
        self.map_update_max_correction_translation = float(
            self.get_parameter("map_update_max_correction_translation").value
        )
        self.map_update_max_correction_rotation_deg = float(
            self.get_parameter("map_update_max_correction_rotation_deg").value
        )
        self.allow_odom_map_update_when_icp_rejected = bool(
            self.get_parameter("allow_odom_map_update_when_icp_rejected").value
        )
        self.odom_map_update_min_translation = float(
            self.get_parameter("odom_map_update_min_translation").value
        )
        self.odom_map_update_min_rotation_deg = float(
            self.get_parameter("odom_map_update_min_rotation_deg").value
        )
        self.odom_map_update_max_fitness = float(
            self.get_parameter("odom_map_update_max_fitness").value
        )
        self.odom_map_update_min_corr = int(self.get_parameter("odom_map_update_min_corr").value)
        self.log_icp_debug = bool(self.get_parameter("log_icp_debug").value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.T_map_odom = np.eye(3, dtype=float)
        self.mto_initialized = False
        self.point_map = np.zeros((0, 2), dtype=float)
        self.last_map_update_base_pose: Optional[np.ndarray] = None
        self.accumulated_scans = deque(maxlen=self.trigger_max_scans)
        self.accumulation_start_base_pose: Optional[np.ndarray] = None
        self.is_turning = False

        self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 10)
        self.create_subscription(Bool, self.is_turning_topic, self.is_turning_callback, 10)
        self.point_map_pub = self.create_publisher(PointCloud2, self.point_map_topic, 10)
        self.stacked_scan_pub = self.create_publisher(PointCloud2, self.stacked_scan_topic, 10)
        self.aligned_scan_pub = self.create_publisher(PointCloud2, self.aligned_scan_topic, 10)

        self.get_logger().info(
            "Open3D clustered novel-fallback point-map localization started | "
            f"scan_topic={self.scan_topic} map_frame={self.map_frame} "
            f"odom_frame={self.odom_frame} base_frame={self.base_frame} "
            f"point_map_topic={self.point_map_topic}"
        )

    def is_turning_callback(self, msg: Bool) -> None:
        was_turning = self.is_turning
        self.is_turning = bool(msg.data)
        if was_turning and not self.is_turning:
            self.accumulated_scans.clear()
            self.accumulation_start_base_pose = None
            self.get_logger().info("Turning ended: cleared Open3D accumulation buffer")

    def try_initialize_mto(self, stamp) -> bool:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.initial_map_to_start_frame,
                stamp,
                timeout=rclpy.time.Duration(seconds=1),
            )
            self.T_map_odom = tfmsg_to_matrix(tf)
            self.mto_initialized = True
            self.publish_map_to_odom(stamp)
            self.get_logger().info(
                f"Initialized {self.map_frame}->{self.odom_frame} from "
                f"{self.map_frame}->{self.initial_map_to_start_frame}"
            )
            return True
        except TransformException as ex:
            # This keeps behavior easy to test: if map->start is not available, start with identity.
            self.T_map_odom = np.eye(3, dtype=float)
            self.mto_initialized = True
            self.publish_map_to_odom(stamp)
            self.get_logger().warn(
                f"Could not initialize {self.map_frame}->{self.initial_map_to_start_frame}; "
                f"using identity {self.map_frame}->{self.odom_frame}: {ex}"
            )
            return True

    def lookup_T(self, target: str, source: str, stamp) -> Optional[np.ndarray]:
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                target,
                source,
                stamp,
                timeout=rclpy.time.Duration(seconds=1),
            )
            return tfmsg_to_matrix(tf_msg)
        except TransformException as ex:
            self.get_logger().warn(f"TF lookup failed {target} <- {source}: {ex}")
            return None

    def publish_map_to_odom(self, stamp) -> None:
        self.tf_broadcaster.sendTransform(
            matrix_to_tf_msg(self.T_map_odom, self.map_frame, self.odom_frame, stamp)
        )

    def publish_debug_clouds(
        self,
        stamp,
        stacked_laser: Optional[np.ndarray] = None,
        T_map_laser_for_stacked: Optional[np.ndarray] = None,
        aligned_map: Optional[np.ndarray] = None,
    ) -> None:
        self.point_map_pub.publish(points_to_pointcloud2(self.point_map, self.map_frame, stamp))

        if stacked_laser is not None and T_map_laser_for_stacked is not None:
            stacked_map = transform_points(T_map_laser_for_stacked, stacked_laser)
            self.stacked_scan_pub.publish(points_to_pointcloud2(stacked_map, self.map_frame, stamp))

        if aligned_map is not None:
            self.aligned_scan_pub.publish(points_to_pointcloud2(aligned_map, self.map_frame, stamp))

    def reset_accumulation(
        self,
        current_points_laser: np.ndarray,
        T_odom_laser: np.ndarray,
        T_odom_base: np.ndarray,
    ) -> None:
        self.accumulated_scans.clear()
        self.accumulated_scans.append(
            BufferedScan(points_laser=current_points_laser.copy(), T_odom_laser=T_odom_laser.copy())
        )
        self.accumulation_start_base_pose = T_odom_base.copy()

    def build_accumulated_points(self, T_odom_laser_current: np.ndarray) -> np.ndarray:
        stacked = []
        T_laser_current_odom = invert_transform(T_odom_laser_current)
        for item in self.accumulated_scans:
            if item.points_laser.shape[0] == 0:
                continue
            T_current_old = T_laser_current_odom @ item.T_odom_laser
            stacked.append(transform_points(T_current_old, item.points_laser))
        return np.vstack(stacked) if stacked else np.zeros((0, 2), dtype=float)

    def motion_trigger_details(self, T_odom_base: np.ndarray) -> Tuple[bool, str, float, float]:
        if self.accumulation_start_base_pose is None:
            return False, "", 0.0, 0.0
        dx, dy, dth = relative_pose(self.accumulation_start_base_pose, T_odom_base)
        trans = math.hypot(dx, dy)
        rot_deg = abs(math.degrees(dth))
        reasons = []
        if trans >= self.trigger_translation:
            reasons.append("distance")
        if rot_deg >= self.trigger_rotation_deg:
            reasons.append("angle")
        if len(self.accumulated_scans) >= self.trigger_max_scans:
            reasons.append("num_scans")
        return len(reasons) > 0, "+".join(reasons), trans, rot_deg

    def filter_significant_clusters(self, points: np.ndarray) -> np.ndarray:
        """Keep scan-ordered clusters that look like real, spatially significant objects.

        The input points are assumed to be in LaserScan angular order. The split
        threshold grows with range so a far wall is not broken into many tiny
        clusters simply because angular beam spacing increases with distance.
        """
        if not self.enable_cluster_filter or points.shape[0] == 0:
            return points.copy()

        clusters = []
        start = 0
        range_norm = max(self.range_max_clip, 1e-6)

        for i in range(1, points.shape[0]):
            r_prev = float(np.linalg.norm(points[i - 1]))
            r_curr = float(np.linalg.norm(points[i]))
            r = 0.5 * (r_prev + r_curr)
            alpha = float(np.clip(r / range_norm, 0.0, 1.0))
            jump_thresh = (
                (1.0 - alpha) * self.cluster_jump_thresh_near
                + alpha * self.cluster_jump_thresh_far
            )
            if float(np.linalg.norm(points[i] - points[i - 1])) > jump_thresh:
                if i - start > 0:
                    clusters.append(points[start:i])
                start = i

        if points.shape[0] - start > 0:
            clusters.append(points[start:])

        kept = []
        for cl in clusters:
            if cl.shape[0] < self.cluster_min_points:
                continue
            mean_range = float(np.mean(np.linalg.norm(cl, axis=1)))
            if mean_range > self.cluster_max_mean_range:
                continue
            extent = float(np.linalg.norm(np.max(cl, axis=0) - np.min(cl, axis=0)))
            if extent < self.cluster_min_extent:
                continue
            kept.append(cl)

        if not kept:
            return np.zeros((0, 2), dtype=float)
        return np.vstack(kept)

    def select_novel_points_for_map(self, points_map: np.ndarray) -> np.ndarray:
        """Return only points that are genuinely new relative to the current map.

        This prevents repeatedly inserting the same wall with slightly different
        poses, which is the main cause of thick/ghost walls. A point is novel if
        its nearest existing map point is farther than novel_point_min_dist.
        Points beyond novel_point_max_dist from the robot/map frame are ignored
        as a simple far-noise guard.
        """
        if points_map.shape[0] == 0:
            return points_map.copy()
        if not self.insert_only_novel_points:
            return points_map.copy()

        ranges = np.linalg.norm(points_map[:, :2], axis=1)
        candidate_mask = ranges <= self.novel_point_max_dist
        candidates = points_map[candidate_mask]
        if candidates.shape[0] == 0:
            return np.zeros((0, 2), dtype=float)

        if self.point_map.shape[0] == 0:
            return candidates.copy()

        # Query nearest existing map point. Open3D's KDTreeFlann operates on 3D
        # point clouds, so convert both clouds to z=0.
        map_cloud = points2d_to_o3d(self.point_map)
        kdtree = o3d.geometry.KDTreeFlann(map_cloud)

        keep = []
        min_dist2 = self.novel_point_min_dist * self.novel_point_min_dist
        for p in candidates:
            query = np.array([float(p[0]), float(p[1]), 0.0], dtype=float)
            _, _, d2 = kdtree.search_knn_vector_3d(query, 1)
            if not d2 or float(d2[0]) >= min_dist2:
                keep.append(p)

        if len(keep) == 0:
            return np.zeros((0, 2), dtype=float)
        return np.asarray(keep, dtype=float).reshape((-1, 2))

    def add_points_to_map(self, points_map: np.ndarray) -> int:
        if points_map.shape[0] == 0:
            return 0
        if self.point_map.shape[0] == 0:
            merged = points_map.copy()
        else:
            merged = np.vstack((self.point_map, points_map))
        before = self.point_map.shape[0]
        merged = voxel_downsample_2d(merged, self.pointmap_voxel)
        merged = limit_points(merged, self.pointmap_max_points)
        self.point_map = merged
        return max(0, int(self.point_map.shape[0] - before))

    def should_update_map(self, T_map_base: np.ndarray) -> bool:
        if self.last_map_update_base_pose is None:
            return True
        dx, dy, dth = relative_pose(self.last_map_update_base_pose, T_map_base)
        return (
            math.hypot(dx, dy) >= self.map_update_min_translation
            or abs(math.degrees(dth)) >= self.map_update_min_rotation_deg
        )

    def should_update_map_with_icp(
        self,
        T_init: np.ndarray,
        result: IcpOutcome,
        T_map_base_smoothed: np.ndarray,
    ) -> bool:
        if result.fitness < self.map_update_min_fitness:
            return False
        if not math.isfinite(result.rmse) or result.rmse > self.map_update_max_rmse:
            return False
        dx, dy, dth = relative_pose(T_init, result.T_map_laser)
        if math.hypot(dx, dy) > self.map_update_max_correction_translation:
            return False
        if abs(math.degrees(dth)) > self.map_update_max_correction_rotation_deg:
            return False
        return self.should_update_map(T_map_base_smoothed)

    def should_update_map_with_odom(self, T_map_base_odom: np.ndarray, result: IcpOutcome) -> bool:
        if not self.allow_odom_map_update_when_icp_rejected:
            return False
        # If fitness is high, ICP probably sees known map but requested a large/ambiguous correction;
        # do not insert odom points in that case, or we may duplicate/blur existing walls.
        if result.fitness > self.odom_map_update_max_fitness:
            return False
        if result.correspondences < self.odom_map_update_min_corr:
            return False
        if self.last_map_update_base_pose is None:
            return True
        dx, dy, dth = relative_pose(self.last_map_update_base_pose, T_map_base_odom)
        return (
            math.hypot(dx, dy) >= self.odom_map_update_min_translation
            or abs(math.degrees(dth)) >= self.odom_map_update_min_rotation_deg
        )

    def run_open3d_icp(
        self,
        source_laser: np.ndarray,
        target_map: np.ndarray,
        T_map_laser_init: np.ndarray,
    ) -> IcpOutcome:
        start = time.time()
        source_ds = voxel_downsample_2d(source_laser, self.open3d_source_voxel)
        target_ds = voxel_downsample_2d(target_map, self.open3d_target_voxel)

        if source_ds.shape[0] < self.min_scan_points or target_ds.shape[0] < self.pointmap_min_points:
            return IcpOutcome(
                T_map_laser=T_map_laser_init.copy(),
                fitness=0.0,
                rmse=float("inf"),
                correspondences=0,
                elapsed_ms=(time.time() - start) * 1000.0,
                converged=False,
            )

        src = points2d_to_o3d(source_ds)
        tgt = points2d_to_o3d(target_ds)
        init4 = matrix_2d_to_4d(T_map_laser_init)

        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max(1, self.open3d_max_iters)
        )
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        result = o3d.pipelines.registration.registration_icp(
            src,
            tgt,
            self.open3d_max_corr_dist,
            init4,
            estimation,
            criteria,
        )

        T2 = matrix_4d_to_2d(np.asarray(result.transformation))
        correspondences = int(np.asarray(result.correspondence_set).shape[0])
        return IcpOutcome(
            T_map_laser=T2,
            fitness=float(result.fitness),
            rmse=float(result.inlier_rmse),
            correspondences=correspondences,
            elapsed_ms=(time.time() - start) * 1000.0,
            converged=correspondences > 0,
        )

    def accept_icp_result(self, T_init: np.ndarray, result: IcpOutcome) -> bool:
        """Gate for using ICP to correct localization / map->odom."""
        if not result.converged:
            return False
        if result.correspondences < self.min_scan_points:
            return False
        if not math.isfinite(result.rmse):
            return False
        if result.fitness < self.open3d_min_fitness:
            return False
        if result.rmse > self.open3d_max_rmse:
            return False

        dx, dy, dth = relative_pose(T_init, result.T_map_laser)
        if math.hypot(dx, dy) > self.icp_accept_max_translation:
            return False
        if abs(math.degrees(dth)) > self.icp_accept_max_rotation_deg:
            return False
        return True

    def accept_map_update_result(self, T_init: np.ndarray, result: IcpOutcome) -> bool:
        """Stricter gate for inserting aligned scan points into the point map."""
        if not self.accept_icp_result(T_init, result):
            return False
        if result.fitness < self.map_update_min_fitness:
            return False
        if result.rmse > self.map_update_max_rmse:
            return False

        dx, dy, dth = relative_pose(T_init, result.T_map_laser)
        if math.hypot(dx, dy) > self.map_update_max_correction_translation:
            return False
        if abs(math.degrees(dth)) > self.map_update_max_correction_rotation_deg:
            return False
        return True

    def scan_callback(self, scan: LaserScan) -> None:
        stamp = scan.header.stamp

        if self.is_turning:
            self.publish_map_to_odom(stamp)
            self.publish_debug_clouds(stamp)
            return

        if not self.mto_initialized:
            self.try_initialize_mto(stamp)
            return

        T_odom_base = self.lookup_T(self.odom_frame, self.base_frame, stamp)
        if T_odom_base is None:
            self.publish_map_to_odom(stamp)
            return

        laser_frame = scan.header.frame_id
        T_base_laser = self.lookup_T(self.laser_base_frame, laser_frame, stamp)
        if T_base_laser is None:
            self.publish_map_to_odom(stamp)
            return

        raw_points_laser = scan_to_points(
            scan,
            self.range_min_clip,
            self.range_max_clip,
            self.scan_stride,
        )
        current_points_laser = self.filter_significant_clusters(raw_points_laser)
        if current_points_laser.shape[0] < self.min_scan_points:
            self.publish_map_to_odom(stamp)
            self.publish_debug_clouds(stamp)
            return

        T_odom_laser = T_odom_base @ T_base_laser
        T_map_laser_init = self.T_map_odom @ T_odom_laser
        T_laser_base = invert_transform(T_base_laser)

        if self.accumulation_start_base_pose is None:
            self.reset_accumulation(current_points_laser, T_odom_laser, T_odom_base)
            self.publish_map_to_odom(stamp)
            self.publish_debug_clouds(stamp, current_points_laser, T_map_laser_init)
            return

        self.accumulated_scans.append(
            BufferedScan(points_laser=current_points_laser.copy(), T_odom_laser=T_odom_laser.copy())
        )
        stacked_laser = self.build_accumulated_points(T_odom_laser)

        # Seed the point map before the first ICP.
        if self.point_map.shape[0] < self.pointmap_min_points:
            seed_map = transform_points(T_map_laser_init, stacked_laser)
            self.add_points_to_map(seed_map)
            T_map_base = T_map_laser_init @ T_laser_base
            self.last_map_update_base_pose = T_map_base.copy()
            self.publish_map_to_odom(stamp)
            self.publish_debug_clouds(stamp, stacked_laser, T_map_laser_init, seed_map)
            self.reset_accumulation(current_points_laser, T_odom_laser, T_odom_base)
            self.get_logger().info(f"Seeded Open3D point map with {self.point_map.shape[0]} points")
            return

        trigger_ready, trigger_reasons, trigger_trans, trigger_rot_deg = self.motion_trigger_details(T_odom_base)
        if not trigger_ready:
            self.publish_map_to_odom(stamp)
            if self.publish_wait_debug:
                self.publish_debug_clouds(stamp, stacked_laser, T_map_laser_init)
            else:
                self.publish_debug_clouds(stamp)
            return

        result = self.run_open3d_icp(stacked_laser, self.point_map, T_map_laser_init)
        loc_accepted = self.accept_icp_result(T_map_laser_init, result)

        aligned_map = transform_points(result.T_map_laser, stacked_laser)
        self.publish_debug_clouds(stamp, stacked_laser, T_map_laser_init, aligned_map)

        map_update_accepted = False
        odom_map_update_accepted = False
        novel_points_for_update = 0

        if loc_accepted:
            T_map_base_est = result.T_map_laser @ T_laser_base
            T_map_odom_est = T_map_base_est @ invert_transform(T_odom_base)
            self.T_map_odom = interpolate_pose(self.T_map_odom, T_map_odom_est, self.pose_smoothing_alpha)

            T_map_laser_smoothed = self.T_map_odom @ T_odom_laser
            T_map_base_smoothed = self.T_map_odom @ T_odom_base

            map_update_accepted = self.should_update_map_with_icp(
                T_map_laser_init,
                result,
                T_map_base_smoothed,
            )
            # Always compute candidate map points after accepted localization.
            # If the strict full map-update gate passes, insert normally.
            # If it fails, we can still insert only novel points, which expands
            # the map without thickening already-known walls.
            new_points_map_all = transform_points(T_map_laser_smoothed, stacked_laser)

            if map_update_accepted:
                new_points_map = self.select_novel_points_for_map(new_points_map_all)
                novel_points_for_update = int(new_points_map.shape[0])
                if (not self.insert_only_novel_points) or new_points_map.shape[0] >= self.novel_min_points:
                    self.add_points_to_map(new_points_map)
                    self.last_map_update_base_pose = T_map_base_smoothed.copy()
                else:
                    map_update_accepted = False
            elif self.allow_novel_update_when_map_update_rejected:
                if (
                    result.fitness >= self.novel_update_min_fitness
                    and math.isfinite(result.rmse)
                    and result.rmse <= self.novel_update_max_rmse
                ):
                    new_points_map = self.select_novel_points_for_map(new_points_map_all)
                    novel_points_for_update = int(new_points_map.shape[0])
                    if new_points_map.shape[0] >= self.novel_min_points:
                        self.add_points_to_map(new_points_map)
                        self.last_map_update_base_pose = T_map_base_smoothed.copy()
        else:
            # Exploration fallback: when ICP fails because we are seeing mostly unmapped space,
            # insert the scan with the odometry prior. Do not correct map->odom in this branch.
            T_map_base_odom = T_map_laser_init @ T_laser_base
            odom_map_update_accepted = self.should_update_map_with_odom(T_map_base_odom, result)
            if odom_map_update_accepted:
                odom_points_map_all = transform_points(T_map_laser_init, stacked_laser)
                odom_points_map = self.select_novel_points_for_map(odom_points_map_all)
                novel_points_for_update = int(odom_points_map.shape[0])
                if (not self.insert_only_novel_points) or odom_points_map.shape[0] >= self.novel_min_points:
                    self.add_points_to_map(odom_points_map)
                    self.last_map_update_base_pose = T_map_base_odom.copy()
                else:
                    odom_map_update_accepted = False

        if self.log_icp_debug:
            dx, dy, dth = relative_pose(T_map_laser_init, result.T_map_laser)
            self.get_logger().info(
                "Open3D ICP | "
                f"reasons={trigger_reasons} dist={trigger_trans:.3f}m rot={trigger_rot_deg:.2f}deg "
                f"fitness={result.fitness:.3f} rmse={result.rmse:.4f} corr={result.correspondences} "
                f"dx={dx:.3f} dy={dy:.3f} dth_deg={math.degrees(dth):.2f} "
                f"time={result.elapsed_ms:.2f}ms "
                f"loc_accepted={loc_accepted} "
                f"map_update_accepted={map_update_accepted} "
                f"odom_map_update_accepted={odom_map_update_accepted} "
                f"filtered_points={current_points_laser.shape[0]} raw_points={raw_points_laser.shape[0]} "
                f"novel_points={novel_points_for_update} "
                f"map_points={self.point_map.shape[0]}"
            )

        self.publish_map_to_odom(stamp)
        self.publish_debug_clouds(stamp, stacked_laser, self.T_map_odom @ T_odom_laser)
        self.reset_accumulation(current_points_laser, T_odom_laser, T_odom_base)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Open3DClusteredPointMapLocalization()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
