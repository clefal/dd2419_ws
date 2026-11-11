#!/usr/bin/env python3

import math
from dataclasses import dataclass
from typing import Optional, Tuple


def wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass(frozen=True)
class FinalApproachCommand:
    left: float
    right: float
    reached: bool = False


class FinalApproachController:
    def __init__(
        self,
        nominal_speed: float,
        turn_gain: float,
        max_angular_speed: float,
        turn_in_place_yaw_thresh: float,
        stop_distance: float,
        min_wheel_duty: float,
    ) -> None:
        self._nominal_speed = float(nominal_speed)
        self._turn_gain = float(turn_gain)
        self._max_angular_speed = float(max_angular_speed)
        self._turn_in_place_yaw_thresh = float(turn_in_place_yaw_thresh)
        self._stop_distance = float(stop_distance)
        self._min_wheel_duty = float(min_wheel_duty)

    def update_gains(
        self,
        nominal_speed: float,
        turn_gain: float,
        max_angular_speed: float,
        turn_in_place_yaw_thresh: float,
        stop_distance: float,
    ) -> None:
        self._nominal_speed = float(nominal_speed)
        self._turn_gain = float(turn_gain)
        self._max_angular_speed = float(max_angular_speed)
        self._turn_in_place_yaw_thresh = float(turn_in_place_yaw_thresh)
        self._stop_distance = float(stop_distance)

    def compute_command(
        self,
        robot_pose: Tuple[float, float, float],
        target_xy: Tuple[float, float],
    ) -> FinalApproachCommand:
        rx, ry, ryaw = robot_pose
        tx, ty = target_xy

        dx = tx - rx
        dy = ty - ry
        dist = math.hypot(dx, dy)
        if dist <= self._stop_distance:
            return FinalApproachCommand(0.0, 0.0, reached=True)

        heading = math.atan2(dy, dx)
        yaw_err = wrap_angle(heading - ryaw)

        cos_y = math.cos(ryaw)
        sin_y = math.sin(ryaw)
        x_r = cos_y * dx + sin_y * dy

        if abs(yaw_err) > self._turn_in_place_yaw_thresh or x_r < 0.0:
            w = clamp(self._turn_gain * yaw_err, -self._max_angular_speed, self._max_angular_speed)
            return FinalApproachCommand(-w, w, reached=False)

        # Simple short-range steering: drive forward and bleed speed close to the stop distance.
        slow_band = max(0.10, 2.0 * self._stop_distance)
        approach = clamp((dist - self._stop_distance) / slow_band, 0.0, 1.0)
        v_min = self._min_wheel_duty + 0.02
        v = v_min + (self._nominal_speed - v_min) * approach

        w_limit = min(self._max_angular_speed, max(0.0, v - self._min_wheel_duty))
        w = clamp(self._turn_gain * yaw_err, -w_limit, w_limit)
        return FinalApproachCommand(v - w, v + w, reached=False)

