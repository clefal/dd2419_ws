#!/usr/bin/env python3

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point, TransformStamped
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
from tf2_ros import TransformException
from visualization_msgs.msg import Marker, MarkerArray


# =========================
# Basic 2D geometry helpers
# =========================

def wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def pose_to_matrix(x: float, y: float, theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([
        [c, -s, x],
        [s,  c, y],
        [0.0, 0.0, 1.0]
    ], dtype=float)


def matrix_to_pose(T: np.ndarray) -> Tuple[float, float, float]:
    x = float(T[0, 2])
    y = float(T[1, 2])
    th = math.atan2(T[1, 0], T[0, 0])
    return x, y, th


def invert_transform(T: np.ndarray) -> np.ndarray:
    R = T[:2, :2]
    t = T[:2, 2]
    Tinv = np.eye(3, dtype=float)
    Tinv[:2, :2] = R.T
    Tinv[:2, 2] = -R.T @ t
    return Tinv


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if pts.shape[0] == 0:
        return pts
    homog = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=float)])
    out = (T @ homog.T).T
    return out[:, :2]


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


def relative_pose(T_from: np.ndarray, T_to: np.ndarray) -> Tuple[float, float, float]:
    dT = invert_transform(T_from) @ T_to
    return matrix_to_pose(dT)


def interpolate_pose(T_old: np.ndarray, T_new: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))

    x0, y0, th0 = matrix_to_pose(T_old)
    x1, y1, th1 = matrix_to_pose(T_new)

    x = (1.0 - alpha) * x0 + alpha * x1
    y = (1.0 - alpha) * y0 + alpha * y1
    dth = wrap_angle(th1 - th0)
    th = wrap_angle(th0 + alpha * dth)

    return pose_to_matrix(x, y, th)


# =========================
# Data structures
# =========================

@dataclass
class LineSegment:
    p1: np.ndarray
    p2: np.ndarray
    q: np.ndarray
    n: np.ndarray
    d: np.ndarray
    length: float
    mid: np.ndarray


@dataclass
class BufferedScan:
    points_laser: np.ndarray
    T_odom_laser: np.ndarray


@dataclass
class IcpResult:
    T: np.ndarray
    num_corr: int
    mean_abs_residual: float
    median_abs_residual: float
    converged: bool
    iterations: int
    time: float


# =========================
# Line segment creation
# =========================

def make_line_segment(p1: np.ndarray, p2: np.ndarray) -> Optional[LineSegment]:
    v = p2 - p1
    length = np.linalg.norm(v)
    if length < 1e-6:
        return None

    d = v / length
    n = np.array([-d[1], d[0]], dtype=float)
    q = 0.5 * (p1 + p2)
    mid = q.copy()

    return LineSegment(
        p1=p1.copy(),
        p2=p2.copy(),
        q=q,
        n=n,
        d=d,
        length=float(length),
        mid=mid
    )


# =========================
# LaserScan preprocessing
# =========================

def scan_to_points(
    scan: LaserScan,
    range_min_clip: float = 0.05,
    range_max_clip: Optional[float] = None,
    stride: int = 1
) -> np.ndarray:
    pts = []
    rmax = scan.range_max if range_max_clip is None else min(scan.range_max, range_max_clip)

    angle = scan.angle_min
    for i, r in enumerate(scan.ranges):
        if i % stride != 0:
            angle += scan.angle_increment
            continue

        if not math.isfinite(r):
            angle += scan.angle_increment
            continue

        if r < max(scan.range_min, range_min_clip) or r > rmax:
            angle += scan.angle_increment
            continue

        pts.append([r * math.cos(angle), r * math.sin(angle)])
        angle += scan.angle_increment

    if len(pts) == 0:
        return np.zeros((0, 2), dtype=float)

    return np.array(pts, dtype=float)

# =========================
# Split-and-merge line extraction
# =========================

def point_line_distance_signed(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    v = b - a
    nv = np.linalg.norm(v)
    if nv < 1e-9:
        return 0.0
    n = np.array([-v[1], v[0]], dtype=float) / nv
    return float(n @ (p - a))


def fit_segment_tls(points: np.ndarray) -> Optional[LineSegment]:
    if points.shape[0] < 2:
        return None

    c = np.mean(points, axis=0)
    X = points - c
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    d = Vt[0]
    d = d / np.linalg.norm(d)

    proj = X @ d
    p1 = c + np.min(proj) * d
    p2 = c + np.max(proj) * d
    return make_line_segment(p1, p2)


def split_and_merge_recursive(points: np.ndarray, split_thresh: float, min_points: int) -> List[np.ndarray]:
    if points.shape[0] < min_points:
        return []

    a = points[0]
    b = points[-1]

    if np.linalg.norm(b - a) < 1e-6:
        return []

    dists = np.array([abs(point_line_distance_signed(p, a, b)) for p in points], dtype=float)
    idx = int(np.argmax(dists))
    max_dist = float(dists[idx])

    if max_dist > split_thresh and idx > 1 and idx < points.shape[0] - 2:
        left = split_and_merge_recursive(points[:idx + 1], split_thresh, min_points)
        right = split_and_merge_recursive(points[idx:], split_thresh, min_points)
        return left + right

    return [points]


def cluster_scan_points(points: np.ndarray, jump_thresh: float = 0.25) -> List[np.ndarray]:
    if points.shape[0] == 0:
        return []

    clusters = []
    start = 0
    for i in range(1, points.shape[0]):
        if np.linalg.norm(points[i] - points[i - 1]) > jump_thresh:
            if i - start >= 2:
                clusters.append(points[start:i])
            start = i

    if points.shape[0] - start >= 2:
        clusters.append(points[start:])

    return clusters


def extract_lines_from_scan(
    points: np.ndarray,
    jump_thresh: float = 0.25,
    split_thresh: float = 0.04,
    min_points: int = 10,
    min_length: float = 0.40
) -> List[LineSegment]:
    clusters = cluster_scan_points(points, jump_thresh=jump_thresh)
    segments: List[LineSegment] = []

    for cl in clusters:
        pieces = split_and_merge_recursive(cl, split_thresh=split_thresh, min_points=min_points)
        for pc in pieces:
            seg = fit_segment_tls(pc)
            if seg is not None and seg.length >= min_length:
                segments.append(seg)

    return segments


# =========================
# Point-to-line association
# =========================

def closest_line_for_point(
    p_map: np.ndarray,
    map_lines: List[LineSegment],
    max_perp_dist: float = 0.12,
    endpoint_margin: float = 0.08,
    local_radius: float = 4.0
) -> Optional[LineSegment]:
    best_line = None
    best_score = float("inf")

    for line in map_lines:
        if np.linalg.norm(line.mid - p_map) > local_radius:
            continue

        rel = p_map - line.p1
        along = float(rel @ line.d)
        perp = abs(float(rel @ line.n))

        if along < -endpoint_margin or along > line.length + endpoint_margin:
            continue
        if perp > max_perp_dist:
            continue

        score = perp
        if score < best_score:
            best_score = score
            best_line = line

    return best_line


def huber_weight(abs_r: float, delta: float) -> float:
    if abs_r <= delta:
        return 1.0
    return float(delta / max(abs_r, 1e-12))


# =========================
# Robust point-to-line ICP
# =========================

def icp_point_to_line_robust(
    points_laser: np.ndarray,
    map_lines: List[LineSegment],
    T_init: np.ndarray,
    max_iters: int = 15,
    min_corr: int = 20,
    max_perp_dist: float = 0.12,
    huber_delta: float = 0.05,
    max_step_translation: float = 0.05,
    max_step_rotation_deg: float = 2.0
) -> IcpResult:
    init_time = time.time()
    if points_laser.shape[0] == 0 or len(map_lines) == 0:
        return IcpResult(
            T=T_init.copy(),
            num_corr=0,
            mean_abs_residual=float("inf"),
            median_abs_residual=float("inf"),
            converged=False,
            iterations=0
        )

    x, y, th = matrix_to_pose(T_init)

    final_num_corr = 0
    final_mean_abs = float("inf")
    final_median_abs = float("inf")
    converged = False
    max_step_rot = math.radians(max_step_rotation_deg)

    for it in range(max_iters):
        c = math.cos(th)
        s = math.sin(th)
        R = np.array([[c, -s], [s, c]], dtype=float)
        t = np.array([x, y], dtype=float)

        J_rows = []
        e_rows = []

        for p in points_laser:
            Rp = R @ p
            p_map = Rp + t

            line = closest_line_for_point(
                p_map,
                map_lines,
                max_perp_dist=max_perp_dist
            )
            if line is None:
                continue

            e = float(line.n @ (p_map - line.q))
            dtheta = np.array([-Rp[1], Rp[0]], dtype=float)
            J = np.array([
                line.n[0],
                line.n[1],
                float(line.n @ dtheta)
            ], dtype=float)

            J_rows.append(J)
            e_rows.append(e)

        final_num_corr = len(J_rows)
        if final_num_corr < min_corr:
            break

        J = np.vstack(J_rows)
        e = np.array(e_rows, dtype=float)

        abs_e = np.abs(e)
        final_mean_abs = float(np.mean(abs_e))
        final_median_abs = float(np.median(abs_e))

        W = np.array([huber_weight(v, huber_delta) for v in abs_e], dtype=float)
        sqrtW = np.sqrt(W)

        Jw = J * sqrtW[:, None]
        ew = e * sqrtW

        H = Jw.T @ Jw
        g = Jw.T @ ew
        H += 1e-6 * np.eye(3)

        try:
            dx = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break

        # Step clamp to avoid violent updates inside one ICP solve
        trans_step = float(np.linalg.norm(dx[:2]))
        rot_step = abs(float(dx[2]))

        if trans_step > max_step_translation:
            dx[:2] *= max_step_translation / max(trans_step, 1e-12)

        if rot_step > max_step_rot:
            dx[2] *= max_step_rot / max(rot_step, 1e-12)

        x += float(dx[0])
        y += float(dx[1])
        th = wrap_angle(th + float(dx[2]))

        if np.linalg.norm(dx[:2]) < 1e-4 and abs(float(dx[2])) < math.radians(0.05):
            converged = True
            break

    return IcpResult(
        T=pose_to_matrix(x, y, th),
        num_corr=final_num_corr,
        mean_abs_residual=final_mean_abs,
        median_abs_residual=final_median_abs,
        converged=converged,
        iterations=it + 1 if max_iters > 0 else 0,
        time=(time.time() - init_time) * 1000
    )

# def icp_point_to_line_robust(
#     points_laser: np.ndarray,
#     map_lines: List[LineSegment],
#     T_init: np.ndarray,
#     max_iters: int = 15,
#     min_corr: int = 20,
#     max_perp_dist: float = 0.12,
#     huber_delta: float = 0.05,
#     max_step_translation: float = 0.05,
#     max_step_rotation_deg: float = 2.0
# ) -> IcpResult:
#     init_time = time.time()
#     if points_laser.shape[0] == 0 or len(map_lines) == 0:
#         return IcpResult(
#             T=T_init.copy(),
#             num_corr=0,
#             mean_abs_residual=float("inf"),
#             median_abs_residual=float("inf"),
#             converged=False,
#             iterations=0
#         )

#     x, y, th = matrix_to_pose(T_init)

#     final_num_corr = 0
#     final_mean_abs = float("inf")
#     final_median_abs = float("inf")
#     converged = False
#     max_step_rot = math.radians(max_step_rotation_deg)

#     # Pre-allocate
#     I3 = np.eye(3, dtype=float)

#     for it in range(max_iters):
#         c = math.cos(th)
#         s = math.sin(th)

#         # Avoid recreating full matrices when possible
#         R = np.array([[c, -s], [s, c]], dtype=float)
#         t = np.array([x, y], dtype=float)

#         # Vectorized transform
#         Rp_all = (R @ points_laser.T).T
#         p_map_all = Rp_all + t

#         J_rows = []
#         e_rows = []

#         for i in range(points_laser.shape[0]):
#             Rp = Rp_all[i]
#             p_map = p_map_all[i]

#             line = closest_line_for_point(
#                 p_map,
#                 map_lines,
#                 max_perp_dist=max_perp_dist
#             )
#             if line is None:
#                 continue

#             # residual
#             e = float(line.n @ (p_map - line.q))

#             # Jacobian
#             dtheta = np.array([-Rp[1], Rp[0]], dtype=float)
#             J = np.array([
#                 line.n[0],
#                 line.n[1],
#                 float(line.n @ dtheta)
#             ], dtype=float)

#             J_rows.append(J)
#             e_rows.append(e)

#         final_num_corr = len(J_rows)
#         if final_num_corr < min_corr:
#             break

#         J = np.vstack(J_rows)
#         e = np.array(e_rows, dtype=float)

#         abs_e = np.abs(e)
#         final_mean_abs = float(np.mean(abs_e))
#         final_median_abs = float(np.median(abs_e))

#         # Vectorized Huber weights
#         W = np.where(
#             abs_e <= huber_delta,
#             1.0,
#             huber_delta / np.maximum(abs_e, 1e-12)
#         )
#         sqrtW = np.sqrt(W)

#         # Weighted least squares
#         Jw = J * sqrtW[:, None]
#         ew = e * sqrtW

#         H = Jw.T @ Jw
#         g = Jw.T @ ew
#         H += 1e-6 * I3  # reuse identity

#         try:
#             dx = -np.linalg.solve(H, g)
#         except np.linalg.LinAlgError:
#             break

#         # Step clamp
#         trans_step = float(np.linalg.norm(dx[:2]))
#         rot_step = abs(float(dx[2]))

#         if trans_step > max_step_translation:
#             dx[:2] *= max_step_translation / max(trans_step, 1e-12)

#         if rot_step > max_step_rot:
#             dx[2] *= max_step_rot / max(rot_step, 1e-12)

#         x += float(dx[0])
#         y += float(dx[1])
#         th = wrap_angle(th + float(dx[2]))

#         # Convergence check
#         if trans_step < 1e-4 and rot_step < math.radians(0.05):
#             converged = True
#             break

#     return IcpResult(
#         T=pose_to_matrix(x, y, th),
#         num_corr=final_num_corr,
#         mean_abs_residual=final_mean_abs,
#         median_abs_residual=final_median_abs,
#         converged=converged,
#         iterations=it + 1 if max_iters > 0 else 0,
#         time=(time.time() - init_time) * 1000
#     )


# =========================
# Main ROS2 node
# =========================

class IcpScanToLine(Node):
    def __init__(self):
        super().__init__("scan_to_line_slam")

        # Topics / frames
        # LaserScan input topic. Default assumes an upstream scan preprocessor node.
        self.declare_parameter("scan_topic", "/localization/preprocessed_scan")
        # Boolean topic indicating when the platform is turning; scans are skipped while true.
        self.declare_parameter("is_turning_topic", "/nav/is_turning")
        # Debug visualization topic for publishing the current map lines.
        self.declare_parameter("map_lines_topic", "/localization/map_lines")
        # Robot base frame used when composing poses.
        self.declare_parameter("base_frame", "base_link_temp")
        # Odometry frame used as the short-term motion prior.
        self.declare_parameter("odom_frame", "odom_temp")  # changed to odom_temp
        # Global frame where the line map is expressed.
        self.declare_parameter("map_frame", "map")

        # Preprocessing
        # Maximum range kept when turning scan beams into points for mapping and ICP.
        self.declare_parameter("range_max_clip", 4.0)
        # Number of consecutive scans stacked together in the current laser frame.
        self.declare_parameter("stack_scans", 4)

        # Line extraction
        # Split ordered points into separate clusters when consecutive points are farther apart than this.
        self.declare_parameter("cluster_jump_thresh", 0.25)
        # Split-and-merge deviation threshold; smaller values produce more, shorter segments.
        self.declare_parameter("split_thresh", 0.05)
        # Minimum number of points required before a candidate segment is accepted as a line.
        self.declare_parameter("line_min_points", 20)
        # Minimum line length required before a detected segment is kept.
        self.declare_parameter("line_min_length", 0.40)

        # ICP
        # Maximum number of scan-to-line ICP iterations per callback.
        self.declare_parameter("icp_max_iters", 15)
        # Minimum number of valid point-to-line correspondences required for acceptance.
        self.declare_parameter("icp_min_corr", 20)
        # Maximum perpendicular point-to-line distance allowed when building correspondences.
        self.declare_parameter("icp_max_perp_dist", 0.20)
        # Huber loss transition point; larger residuals are down-weighted.
        self.declare_parameter("icp_huber_delta", 0.05)
        # Maximum translation correction allowed relative to the odometry-based initial guess.
        self.declare_parameter("icp_accept_max_translation", 0.20)
        # Maximum rotation correction allowed relative to the odometry-based initial guess.
        self.declare_parameter("icp_accept_max_rotation_deg", 10.0)
        # Reject ICP if the median absolute residual is above this threshold.
        self.declare_parameter("icp_accept_max_median_residual", 0.08)
        # Reject ICP if the mean absolute residual is above this threshold.
        self.declare_parameter("icp_accept_max_mean_residual", 0.10)

        # Smoothing
        # Low-pass factor used when updating map->odom from the accepted ICP estimate.
        self.declare_parameter("pose_smoothing_alpha", 0.20)

        # Map maintenance
        # Hard cap on the number of stored map line segments.
        self.declare_parameter("map_max_lines", 50)
        # Minimum midpoint separation before a similar detected line is inserted into the map.
        self.declare_parameter("map_insert_min_separation", 0.35)
        # If true, merge near-duplicate collinear segments into a longer segment.
        self.declare_parameter("map_merge_lines", True)
        # Maximum orientation difference (deg) for merging collinear segments.
        self.declare_parameter("map_merge_angle_deg", 5.0)
        # Maximum perpendicular distance (m) between segments for merging.
        self.declare_parameter("map_merge_perp_dist", 0.1)
        # Maximum allowed along-line gap (m) between segment intervals for merging.
        self.declare_parameter("map_merge_max_gap", 0.25)
        # Minimum number of stored lines before the node switches from map seeding to ICP tracking.
        self.declare_parameter("init_min_lines", 2)
        # Minimum base translation required before adding more lines to the map.
        self.declare_parameter("map_update_min_translation", 0.15)
        # Minimum base rotation required before adding more lines to the map.
        self.declare_parameter("map_update_min_rotation_deg", 8.0)

        # Debug
        # Enable per-scan ICP logging with residual and correction information.
        self.declare_parameter("log_icp_debug", True)

        # Read params
        self.scan_topic = self.get_parameter("scan_topic").value
        self.is_turning_topic = self.get_parameter("is_turning_topic").value
        self.map_lines_topic = self.get_parameter("map_lines_topic").value
        self.base_frame = self.get_parameter("base_frame").value
        self.odom_frame = self.get_parameter("odom_frame").value
        self.map_frame = self.get_parameter("map_frame").value

        self.range_max_clip = float(self.get_parameter("range_max_clip").value)
        self.stack_scans = max(1, int(self.get_parameter("stack_scans").value))

        self.cluster_jump_thresh = float(self.get_parameter("cluster_jump_thresh").value)
        self.split_thresh = float(self.get_parameter("split_thresh").value)
        self.line_min_points = int(self.get_parameter("line_min_points").value)
        self.line_min_length = float(self.get_parameter("line_min_length").value)

        self.icp_max_iters = int(self.get_parameter("icp_max_iters").value)
        self.icp_min_corr = int(self.get_parameter("icp_min_corr").value)
        self.icp_max_perp_dist = float(self.get_parameter("icp_max_perp_dist").value)
        self.icp_huber_delta = float(self.get_parameter("icp_huber_delta").value)
        self.icp_accept_max_translation = float(self.get_parameter("icp_accept_max_translation").value)
        self.icp_accept_max_rotation_deg = float(self.get_parameter("icp_accept_max_rotation_deg").value)
        self.icp_accept_max_median_residual = float(self.get_parameter("icp_accept_max_median_residual").value)
        self.icp_accept_max_mean_residual = float(self.get_parameter("icp_accept_max_mean_residual").value)

        self.pose_smoothing_alpha = float(self.get_parameter("pose_smoothing_alpha").value)

        self.map_max_lines = int(self.get_parameter("map_max_lines").value)
        self.map_insert_min_separation = float(self.get_parameter("map_insert_min_separation").value)
        self.map_merge_lines = bool(self.get_parameter("map_merge_lines").value)
        self.map_merge_angle_deg = float(self.get_parameter("map_merge_angle_deg").value)
        self.map_merge_perp_dist = float(self.get_parameter("map_merge_perp_dist").value)
        self.map_merge_max_gap = float(self.get_parameter("map_merge_max_gap").value)
        self.init_min_lines = int(self.get_parameter("init_min_lines").value)
        self.map_update_min_translation = float(self.get_parameter("map_update_min_translation").value)
        self.map_update_min_rotation_deg = float(self.get_parameter("map_update_min_rotation_deg").value)

        self.log_icp_debug = bool(self.get_parameter("log_icp_debug").value)

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # State
        self.T_map_odom = np.eye(3, dtype=float)
        self.mto_initialized = False

        self.map_lines: List[LineSegment] = []
        self.initialized = False

        self.scan_buffer = deque(maxlen=max(0, self.stack_scans - 1))
        self.is_turning = False

        self.last_map_update_base_pose: Optional[np.ndarray] = None

        # IO
        self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 10)
        self.create_subscription(Bool, self.is_turning_topic, self.is_turning_callback, 10)
        self.map_lines_pub = self.create_publisher(MarkerArray, self.map_lines_topic, 10)

        self.get_logger().info(
            f"scan_to_line_slam started | scan_topic={self.scan_topic}, "
            f"stack_scans={self.stack_scans}, "
            f"smoothing_alpha={self.pose_smoothing_alpha:.2f}"
        )

    def is_turning_callback(self, msg: Bool) -> None:
        was_turning = self.is_turning
        self.is_turning = bool(msg.data)

        if was_turning and not self.is_turning:
            self.scan_buffer.clear()
            self.get_logger().info("Turning ended: cleared stacked scan buffer")

    def try_initialize_mto(self, stamp) -> bool:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame,
                "start",
                stamp,
                timeout=rclpy.time.Duration(seconds=1)
            )
            self.T_map_odom = tfmsg_to_matrix(tf)
            self.mto_initialized = True
            self.get_logger().info(f"Initialized {self.map_frame}->{self.odom_frame} from {self.map_frame}->start")

            tf_map_odom = TransformStamped()
            tf_map_odom.header.stamp = tf.header.stamp
            tf_map_odom.header.frame_id = self.map_frame
            tf_map_odom.child_frame_id = self.odom_frame
            tf_map_odom.transform = tf.transform
            self.tf_broadcaster.sendTransform(tf_map_odom)
            return True
        except TransformException as ex:
            self.T_map_odom = np.eye(3, dtype=float)
            self.mto_initialized = False
            self.get_logger().warn(f"Could not initialize map->odom from TF, using identity: {ex}")
            return False

    def lookup_T(self, target: str, source: str, stamp) -> Optional[np.ndarray]:
        try:
            # future = self.tf_buffer.wait_for_transform_async(target, source, stamp)
            # rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
            tf_msg = self.tf_buffer.lookup_transform(
                target, source, stamp, timeout=rclpy.time.Duration(seconds=1)
            )
            return tfmsg_to_matrix(tf_msg)
        except TransformException as ex:
            self.get_logger().warn(f"TF lookup failed {target} <- {source}: {ex}")
            return None

    def preprocess_scan(self, scan: LaserScan) -> Tuple[LaserScan, np.ndarray]:
        # Scan-level filtering is done upstream (see `filter_scan` node). Here we only
        # convert the already preprocessed scan into points.
        points = scan_to_points(
            scan,
            range_max_clip=self.range_max_clip,
        )
        return scan, points

    def build_stacked_points(self, current_points_laser: np.ndarray, T_odom_laser_current: np.ndarray) -> np.ndarray:
        stacked = [current_points_laser]
        T_laser_current_odom = invert_transform(T_odom_laser_current)

        for item in self.scan_buffer:
            if item.points_laser.shape[0] == 0:
                continue
            T_current_old = T_laser_current_odom @ item.T_odom_laser
            pts_in_current = transform_points(T_current_old, item.points_laser)
            stacked.append(pts_in_current)

        return np.vstack(stacked) if len(stacked) > 0 else np.zeros((0, 2), dtype=float)

    def push_scan_buffer(self, points_laser: np.ndarray, T_odom_laser: np.ndarray) -> None:
        if self.scan_buffer.maxlen == 0:
            return
        self.scan_buffer.append(
            BufferedScan(
                points_laser=points_laser.copy(),
                T_odom_laser=T_odom_laser.copy()
            )
        )

    def should_insert_line(self, line: LineSegment) -> bool:
        for existing in self.map_lines:
            angle = math.acos(np.clip(abs(existing.d @ line.d), 0.0, 1.0))
            if angle > math.radians(10.0):
                continue

            perp_dist = abs(existing.n @ (line.mid - existing.q))
            if perp_dist > 0.15:
                continue

            if np.linalg.norm(existing.mid - line.mid) < self.map_insert_min_separation:
                return False

        return True

    @staticmethod
    def _interval_along(line: LineSegment, origin: np.ndarray, d: np.ndarray) -> Tuple[float, float]:
        s1 = float(d @ (line.p1 - origin))
        s2 = float(d @ (line.p2 - origin))
        return (min(s1, s2), max(s1, s2))

    def _lines_mergeable(self, a: LineSegment, b: LineSegment) -> bool:
        # Collinear-ish check (direction + perpendicular offset), then ensure the
        # segments touch/overlap along the line up to a small gap.
        angle = math.acos(np.clip(abs(float(a.d @ b.d)), 0.0, 1.0))
        if angle > math.radians(self.map_merge_angle_deg):
            return False

        perp_ab = abs(float(a.n @ (b.mid - a.q)))
        perp_ba = abs(float(b.n @ (a.mid - b.q)))
        if perp_ab > self.map_merge_perp_dist or perp_ba > self.map_merge_perp_dist:
            return False

        d = a.d
        origin = a.q
        ia = self._interval_along(a, origin, d)
        ib = self._interval_along(b, origin, d)
        gap = max(0.0, max(ia[0], ib[0]) - min(ia[1], ib[1]))
        return gap <= self.map_merge_max_gap

    def _merge_line_into_map(self, line: LineSegment) -> bool:
        """
        Merge `line` with any existing map segments that are near-duplicate
        (collinear, small perpendicular offset, small gap). Returns True if a
        merge happened (and the map was updated), False otherwise.
        """
        if not self.map_merge_lines or len(self.map_lines) == 0:
            return False

        merged_pts = [line.p1, line.p2]
        merged = False
        candidate = line

        # Iteratively merge: once the segment grows, it may become mergeable
        # with segments that previously were just outside the interval.
        while True:
            merge_indices = []
            for idx, existing in enumerate(self.map_lines):
                if self._lines_mergeable(existing, candidate):
                    merge_indices.append(idx)

            if not merge_indices:
                break

            for idx in reversed(merge_indices):
                ex = self.map_lines.pop(idx)
                merged_pts.append(ex.p1)
                merged_pts.append(ex.p2)
                merged = True

            merged_seg = fit_segment_tls(np.vstack(merged_pts))
            if merged_seg is None:
                break
            candidate = merged_seg

        if merged:
            self.map_lines.append(candidate)
        return merged

    def insert_lines_into_map(self, lines_map: List[LineSegment]) -> None:
        for line in lines_map:
            if self._merge_line_into_map(line):
                continue
            if self.should_insert_line(line):
                self.map_lines.append(line)

        if len(self.map_lines) > self.map_max_lines:
            self.map_lines = self.map_lines[-self.map_max_lines:]

    def publish_map_lines_markers(self, stamp) -> None:
        markers = MarkerArray()

        delete_marker = Marker()
        delete_marker.header.frame_id = self.map_frame
        delete_marker.header.stamp = stamp
        delete_marker.ns = "map_lines"
        delete_marker.action = Marker.DELETEALL
        markers.markers.append(delete_marker)

        for idx, line in enumerate(self.map_lines):
            marker = Marker()
            marker.header.frame_id = self.map_frame
            marker.header.stamp = stamp
            marker.ns = "map_lines"
            marker.id = idx
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            marker.scale.x = 0.03
            marker.color.r = 0.1
            marker.color.g = 0.9
            marker.color.b = 0.2
            marker.color.a = 1.0

            p1 = Point()
            p1.x = float(line.p1[0])
            p1.y = float(line.p1[1])
            p1.z = 0.0

            p2 = Point()
            p2.x = float(line.p2[0])
            p2.y = float(line.p2[1])
            p2.z = 0.0

            marker.points = [p1, p2]
            markers.markers.append(marker)

        self.map_lines_pub.publish(markers)

    def seed_map_from_scan(self, points_laser: np.ndarray, T_map_laser: np.ndarray) -> None:
        raw_lines = extract_lines_from_scan(
            points_laser,
            jump_thresh=self.cluster_jump_thresh,
            split_thresh=self.split_thresh,
            min_points=self.line_min_points,
            min_length=self.line_min_length
        )

        lines_map = []
        for seg in raw_lines:
            p1m = transform_points(T_map_laser, seg.p1.reshape(1, 2))[0]
            p2m = transform_points(T_map_laser, seg.p2.reshape(1, 2))[0]
            mapped = make_line_segment(p1m, p2m)
            if mapped is not None:
                lines_map.append(mapped)

        self.insert_lines_into_map(lines_map)

    def should_update_map(self, T_map_base: np.ndarray) -> bool:
        if self.last_map_update_base_pose is None:
            return True

        dx, dy, dth = relative_pose(self.last_map_update_base_pose, T_map_base)
        trans = math.hypot(dx, dy)
        rot = abs(dth)

        return (
            trans >= self.map_update_min_translation or
            rot >= math.radians(self.map_update_min_rotation_deg)
        )

    def accept_icp_result(self, T_init: np.ndarray, result: IcpResult) -> bool:
        if result.num_corr < self.icp_min_corr:
            return False

        if not math.isfinite(result.mean_abs_residual) or not math.isfinite(result.median_abs_residual):
            return False

        if result.median_abs_residual > self.icp_accept_max_median_residual:
            return False

        if result.mean_abs_residual > self.icp_accept_max_mean_residual:
            return False

        dx, dy, dth = relative_pose(T_init, result.T)
        trans = math.hypot(dx, dy)
        rot_deg = abs(math.degrees(dth))

        if trans > self.icp_accept_max_translation:
            return False

        if rot_deg > self.icp_accept_max_rotation_deg:
            return False

        return True

    def publish_map_to_odom(self, stamp) -> None:
        tf_msg = matrix_to_tf_msg(self.T_map_odom, self.map_frame, self.odom_frame, stamp)
        self.tf_broadcaster.sendTransform(tf_msg)

    def scan_callback(self, scan: LaserScan) -> None:
        if self.is_turning:
            self.get_logger().warn("Ignoring scan while turning")
            return

        stamp = scan.header.stamp

        if not self.mto_initialized:
            self.try_initialize_mto(stamp)
            return

        T_odom_base = self.lookup_T(self.odom_frame, self.base_frame, stamp)
        if T_odom_base is None:
            return

        laser_frame = scan.header.frame_id
        T_base_laser = self.lookup_T("base_link", laser_frame, stamp)
        if T_base_laser is None:
            return

        T_odom_laser = T_odom_base @ T_base_laser

        _, current_points_laser = self.preprocess_scan(scan)
        if current_points_laser.shape[0] < 20:
            return

        stacked_points_laser = self.build_stacked_points(current_points_laser, T_odom_laser)
        if stacked_points_laser.shape[0] < 20:
            return

        T_map_laser_init = self.T_map_odom @ T_odom_laser

        # Initialization
        if not self.initialized or len(self.map_lines) < self.init_min_lines:
            self.seed_map_from_scan(stacked_points_laser, T_map_laser_init)
            self.initialized = True

            T_laser_base = invert_transform(T_base_laser)
            self.last_map_update_base_pose = T_map_laser_init @ T_laser_base

            self.publish_map_to_odom(stamp)
            self.publish_map_lines_markers(stamp)
            self.push_scan_buffer(current_points_laser, T_odom_laser)

            self.get_logger().info(f"Map initialized with {len(self.map_lines)} lines")
            return

        # Robust ICP
        result = icp_point_to_line_robust(
            stacked_points_laser,
            self.map_lines,
            T_map_laser_init,
            max_iters=self.icp_max_iters,
            min_corr=self.icp_min_corr,
            max_perp_dist=self.icp_max_perp_dist,
            huber_delta=self.icp_huber_delta,
            max_step_translation=self.icp_accept_max_translation,
            max_step_rotation_deg=self.icp_accept_max_rotation_deg
        )

        accepted = self.accept_icp_result(T_map_laser_init, result)

        if self.log_icp_debug:
            dx, dy, dth = relative_pose(T_map_laser_init, result.T)
            self.get_logger().info(
                f"ICP | corr={result.num_corr} "
                f"iters={result.iterations} "
                f"mean={result.mean_abs_residual:.4f} "
                f"median={result.median_abs_residual:.4f} "
                f"dx={dx:.3f} dy={dy:.3f} dth_deg={math.degrees(dth):.2f} "
                f"time={result.time:.3f}ms "
                f"accepted={accepted}"
            )

        if accepted:
            T_laser_base = invert_transform(T_base_laser)
            T_map_base_est = result.T @ T_laser_base
            T_map_odom_est = T_map_base_est @ invert_transform(T_odom_base)

            # Smooth map->odom instead of snapping to raw ICP result
            self.T_map_odom = interpolate_pose(
                self.T_map_odom,
                T_map_odom_est,
                self.pose_smoothing_alpha
            )

            # Optional map update
            T_map_laser_smoothed = self.T_map_odom @ T_odom_laser
            T_map_base_smoothed = self.T_map_odom @ T_odom_base

            if self.should_update_map(T_map_base_smoothed):
                raw_lines = extract_lines_from_scan(
                    stacked_points_laser,
                    jump_thresh=self.cluster_jump_thresh,
                    split_thresh=self.split_thresh,
                    min_points=self.line_min_points,
                    min_length=self.line_min_length
                )

                lines_map = []
                for seg in raw_lines:
                    p1m = transform_points(T_map_laser_smoothed, seg.p1.reshape(1, 2))[0]
                    p2m = transform_points(T_map_laser_smoothed, seg.p2.reshape(1, 2))[0]
                    mapped = make_line_segment(p1m, p2m)
                    if mapped is not None:
                        lines_map.append(mapped)

                self.insert_lines_into_map(lines_map)
                self.last_map_update_base_pose = T_map_base_smoothed

        else:
            # Reject frame, keep previous map->odom
            pass

        self.publish_map_to_odom(stamp)
        self.publish_map_lines_markers(stamp)
        self.push_scan_buffer(current_points_laser, T_odom_laser)


def main(args=None):
    rclpy.init(args=args)
    node = IcpScanToLine()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
