#!/usr/bin/env python3

import math
import time
from collections import deque
from typing import Optional

import numpy as np
import rclpy

from sensor_msgs.msg import LaserScan

from localization.icp_scan_to_line import (
    BufferedScan,
    IcpScanToLine,
    LineSegment,
    extract_lines_from_scan,
    icp_point_to_line_robust,
    interpolate_pose,
    invert_transform,
    make_line_segment,
    relative_pose,
    transform_points,
)


class IcpScanToLineMotionTriggered(IcpScanToLine):
    def __init__(self):
        super().__init__()

        self.declare_parameter("trigger_translation", 0.25)
        self.declare_parameter("trigger_rotation_deg", 10.0)
        self.declare_parameter("trigger_max_scans", 14)
        self.declare_parameter("init_seed_min_lines", 3)
        self.declare_parameter("init_min_angle_separation_deg", 30.0)
        self.declare_parameter("publish_wait_debug", False)

        self.trigger_translation = float(self.get_parameter("trigger_translation").value)
        self.trigger_rotation_deg = float(self.get_parameter("trigger_rotation_deg").value)
        self.trigger_max_scans = max(1, int(self.get_parameter("trigger_max_scans").value))
        self.init_seed_min_lines = max(1, int(self.get_parameter("init_seed_min_lines").value))
        self.init_min_angle_separation_deg = float(
            self.get_parameter("init_min_angle_separation_deg").value
        )
        self.publish_wait_debug = bool(self.get_parameter("publish_wait_debug").value)

        self.accumulated_scans = deque(maxlen=self.trigger_max_scans)
        self.accumulation_start_base_pose: Optional[np.ndarray] = None

        self.get_logger().info(
            "motion-triggered ICP enabled | "
            f"trigger_translation={self.trigger_translation:.2f}m "
            f"trigger_rotation_deg={self.trigger_rotation_deg:.1f} "
            f"trigger_max_scans={self.trigger_max_scans} "
            f"init_seed_min_lines={self.init_seed_min_lines} "
            f"init_min_angle_separation_deg={self.init_min_angle_separation_deg:.1f}"
        )

    def is_turning_callback(self, msg) -> None:
        was_turning = self.is_turning
        super().is_turning_callback(msg)
        if was_turning and not self.is_turning:
            self.accumulated_scans.clear()
            self.accumulation_start_base_pose = None

    def build_accumulated_points(
        self,
        T_odom_laser_current: np.ndarray,
    ) -> np.ndarray:
        stacked = []
        T_laser_current_odom = invert_transform(T_odom_laser_current)

        for item in self.accumulated_scans:
            if item.points_laser.shape[0] == 0:
                continue
            T_current_old = T_laser_current_odom @ item.T_odom_laser
            stacked.append(transform_points(T_current_old, item.points_laser))

        return np.vstack(stacked) if stacked else np.zeros((0, 2), dtype=float)

    def reset_accumulation(
        self,
        current_points_laser: np.ndarray,
        T_odom_laser: np.ndarray,
        T_odom_base: np.ndarray,
    ) -> None:
        self.accumulated_scans.clear()
        self.accumulated_scans.append(
            BufferedScan(
                points_laser=current_points_laser.copy(),
                T_odom_laser=T_odom_laser.copy(),
            )
        )
        self.accumulation_start_base_pose = T_odom_base.copy()

    def continue_accumulation_after_failed_init(self, T_odom_base: np.ndarray) -> None:
        # Keep the scans we have already accumulated and start a new motion interval
        # from the current pose. This avoids throwing away partial scene coverage when
        # initialization geometry is still too weak, while still requiring additional
        # motion before retrying initialization.
        self.accumulation_start_base_pose = T_odom_base.copy()

    def motion_trigger_ready(self, T_odom_base: np.ndarray) -> bool:
        if self.accumulation_start_base_pose is None:
            return False

        dx, dy, dth = relative_pose(self.accumulation_start_base_pose, T_odom_base)
        trans = math.hypot(dx, dy)
        rot = abs(math.degrees(dth))

        return (
            trans >= self.trigger_translation
            or rot >= self.trigger_rotation_deg
            or len(self.accumulated_scans) >= self.trigger_max_scans
        )

    def motion_trigger_details(self, T_odom_base: np.ndarray) -> tuple[bool, list[str], float, float]:
        if self.accumulation_start_base_pose is None:
            return False, [], 0.0, 0.0

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

        return len(reasons) > 0, reasons, trans, rot_deg

    def build_lines_from_scan(self, points_laser: np.ndarray, T_map_laser: np.ndarray) -> list[LineSegment]:
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

        return lines_map

    def has_initialization_geometry(self, lines_map: list[LineSegment]) -> bool:
        if len(lines_map) < self.init_seed_min_lines:
            return False

        min_angle = math.radians(self.init_min_angle_separation_deg)
        for i in range(len(lines_map)):
            for j in range(i + 1, len(lines_map)):
                angle = math.acos(np.clip(abs(float(lines_map[i].d @ lines_map[j].d)), 0.0, 1.0))
                if angle >= min_angle:
                    return True

        return False

    def publish_stacked_points_in_map(
        self,
        points_laser: np.ndarray,
        T_map_laser: np.ndarray,
        stamp,
    ) -> None:
        points_map = transform_points(T_map_laser, points_laser)
        self.publish_stacked_points(points_map, self.map_frame, stamp)

    def scan_callback(self, scan: LaserScan) -> None:
        init_time = time.time()
        stamp = scan.header.stamp
        if self.is_turning:
            self.get_logger().warn("Ignoring scan while turning")
            if self.mto_initialized:
                self.publish_map_to_odom(stamp)
            return

        if not self.mto_initialized:
            self.try_initialize_mto(stamp)
            if self.mto_initialized:
                self.publish_map_to_odom(stamp)
            return

        T_odom_base = self.lookup_T(self.odom_frame, self.base_frame, stamp)
        if T_odom_base is None:
            self.publish_map_to_odom(stamp)
            return

        laser_frame = scan.header.frame_id
        T_base_laser = self.lookup_T(self.laser_mount_frame, laser_frame, stamp)
        if T_base_laser is None:
            self.publish_map_to_odom(stamp)
            return

        T_odom_laser = T_odom_base @ T_base_laser

        _, current_points_laser = self.preprocess_scan(scan)
        if current_points_laser.shape[0] < 20:
            self.publish_map_to_odom(stamp)
            return

        T_map_laser_init = self.T_map_odom @ T_odom_laser

        if self.accumulation_start_base_pose is None:
            self.reset_accumulation(current_points_laser, T_odom_laser, T_odom_base)
            self.publish_map_lines_markers(stamp)
            self.publish_map_to_odom(stamp)
            return

        self.accumulated_scans.append(
            BufferedScan(
                points_laser=current_points_laser.copy(),
                T_odom_laser=T_odom_laser.copy(),
            )
        )

        trigger_ready, trigger_reasons, trigger_trans, trigger_rot_deg = self.motion_trigger_details(T_odom_base)
        if not trigger_ready:
            if self.publish_wait_debug:
                stacked_wait = self.build_accumulated_points(T_odom_laser)
                self.publish_stacked_points_in_map(stacked_wait, T_map_laser_init, stamp)
            self.publish_map_lines_markers(stamp)
            self.publish_map_to_odom(stamp)
            return

        stacked_points_laser = self.build_accumulated_points(T_odom_laser)
        if stacked_points_laser.shape[0] < 20:
            self.reset_accumulation(current_points_laser, T_odom_laser, T_odom_base)
            self.publish_map_to_odom(stamp)
            return

        if not self.initialized or len(self.map_lines) < self.init_min_lines:
            scans_used_for_init = len(self.accumulated_scans)
            self.publish_stacked_points_in_map(stacked_points_laser, T_map_laser_init, stamp)
            candidate_lines = self.build_lines_from_scan(stacked_points_laser, T_map_laser_init)
            if not self.has_initialization_geometry(candidate_lines):
                self.get_logger().info(
                    "Initialization waiting for richer geometry | "
                    f"accumulated_scans={scans_used_for_init} "
                    f"candidate_lines={len(candidate_lines)} "
                    f"required_lines={self.init_seed_min_lines} "
                    f"min_angle_sep_deg={self.init_min_angle_separation_deg:.1f}"
                )
                self.continue_accumulation_after_failed_init(T_odom_base)
                self.publish_map_to_odom(stamp)
                self.publish_map_lines_markers(stamp)
                return

            self.insert_lines_into_map(candidate_lines)
            self.initialized = True

            T_laser_base = invert_transform(T_base_laser)
            self.last_map_update_base_pose = T_map_laser_init @ T_laser_base

            self.publish_map_to_odom(stamp)
            self.publish_map_lines_markers(stamp)
            self.reset_accumulation(current_points_laser, T_odom_laser, T_odom_base)

            self.get_logger().info(
                f"Map initialized with {len(self.map_lines)} lines "
                f"from {scans_used_for_init} accumulated scans"
            )
            return

        self.get_logger().info(
            "Running ICP | "
            f"reasons={'+'.join(trigger_reasons)} "
            f"distance={trigger_trans:.3f}m "
            f"angle={trigger_rot_deg:.2f}deg "
            f"accumulated_scans={len(self.accumulated_scans)}"
        )

        result = icp_point_to_line_robust(
            stacked_points_laser,
            self.map_lines,
            T_map_laser_init,
            max_iters=self.icp_max_iters,
            min_corr=self.icp_min_corr,
            max_perp_dist=self.icp_max_perp_dist,
            huber_delta=self.icp_huber_delta,
            max_step_translation=self.icp_accept_max_translation,
            max_step_rotation_deg=self.icp_accept_max_rotation_deg,
            line_cache=self.line_map_cache,
        )
        self.publish_stacked_points_in_map(stacked_points_laser, result.T, stamp)

        accepted = self.accept_icp_result(T_map_laser_init, result)

        if self.log_icp_debug:
            dx, dy, dth = relative_pose(T_map_laser_init, result.T)
            self.get_logger().info(
                f"ICP motion-triggered | corr={result.num_corr} "
                f"iters={result.iterations} "
                f"mean={result.mean_abs_residual:.4f} "
                f"median={result.median_abs_residual:.4f} "
                f"dx={dx:.3f} dy={dy:.3f} dth_deg={math.degrees(dth):.2f} "
                f"time={result.time:.3f}ms "
                f"accepted={accepted} "
                f"accumulated_scans={len(self.accumulated_scans)}"
            )

        if accepted:
            T_laser_base = invert_transform(T_base_laser)
            T_map_base_est = result.T @ T_laser_base
            T_map_odom_est = T_map_base_est @ invert_transform(T_odom_base)

            self.T_map_odom = interpolate_pose(
                self.T_map_odom,
                T_map_odom_est,
                self.pose_smoothing_alpha
            )

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

        finish_time = time.time()
        self.get_logger().info(f"ICP time: {(finish_time - init_time) * 1000.0:.2f}ms")

        self.publish_map_to_odom(stamp)
        self.publish_map_lines_markers(stamp)
        self.reset_accumulation(current_points_laser, T_odom_laser, T_odom_base)


def main(args=None):
    rclpy.init(args=args)
    node = IcpScanToLineMotionTriggered()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
