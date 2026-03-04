#!/usr/bin/env python3

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from nav_msgs.msg import OccupancyGrid


@dataclass(frozen=True)
class GridMeta:
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float


class RandomWaypointExplorer:
    def __init__(
        self,
        min_step_m: float = 1.0,
        max_step_m: float = 2.0,
        min_revisit_dist_m: float = 0.8,
        failed_blacklist_radius_m: float = 0.6,
        occ_lethal: int = 90,
        interior_bias_count: int = 2,
        interior_margin_m: float = 0.4,
        max_samples: int = 500,
        seed: Optional[int] = None,
    ) -> None:
        self._min_step_m = float(min_step_m)
        self._max_step_m = float(max_step_m)
        self._min_revisit_dist_m = float(min_revisit_dist_m)
        self._failed_blacklist_radius_m = float(failed_blacklist_radius_m)
        self._occ_lethal = int(occ_lethal)
        self._interior_bias_count = int(interior_bias_count)
        self._interior_margin_m = float(interior_margin_m)
        self._max_samples = int(max_samples)
        self._rng = random.Random(seed)

        self._workspace_polygon: List[Tuple[float, float]] = []
        self._visited: List[Tuple[float, float]] = []
        self._failed: List[Tuple[float, float]] = []

        self._planning_data: Optional[List[int]] = None
        self._meta: Optional[GridMeta] = None

    def set_workspace_polygon(self, polygon_xy: List[Tuple[float, float]]) -> None:
        self._workspace_polygon = list(polygon_xy)

    def set_planning_grid(self, msg: OccupancyGrid) -> None:
        self._planning_data = list(msg.data)
        self._meta = GridMeta(
            width=msg.info.width,
            height=msg.info.height,
            resolution=msg.info.resolution,
            origin_x=msg.info.origin.position.x,
            origin_y=msg.info.origin.position.y,
        )

    def note_waypoint_dispatched(self, xy: Tuple[float, float]) -> None:
        self._visited.append((float(xy[0]), float(xy[1])))

    def note_waypoint_result(self, xy: Tuple[float, float], status: str) -> None:
        if status == "FAILED":
            self._failed.append((float(xy[0]), float(xy[1])))

    def next_waypoint(self, robot_xy: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        if len(self._workspace_polygon) < 3:
            return None

        rx, ry = robot_xy
        xs = [p[0] for p in self._workspace_polygon]
        ys = [p[1] for p in self._workspace_polygon]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        bounds = (min_x, max_x, min_y, max_y)

        for _ in range(self._max_samples):
            x = self._rng.uniform(min_x, max_x)
            y = self._rng.uniform(min_y, max_y)

            if not self._point_in_polygon(x, y, self._workspace_polygon):
                continue

            d = math.hypot(x - rx, y - ry)
            if d < self._min_step_m or d > self._max_step_m:
                continue

            if not self._is_interior_enough(x, y, bounds):
                continue

            if self._is_near_any((x, y), self._visited, self._min_revisit_dist_m):
                continue

            if self._is_near_any((x, y), self._failed, self._failed_blacklist_radius_m):
                continue

            if not self._is_traversable(x, y):
                continue

            return (x, y)

        # Relax revisit requirement, but keep step range and traversability.
        relaxed_revisit = 0.35 * self._min_revisit_dist_m
        for _ in range(max(40, self._max_samples // 4)):
            x = self._rng.uniform(min_x, max_x)
            y = self._rng.uniform(min_y, max_y)
            if not self._point_in_polygon(x, y, self._workspace_polygon):
                continue
            d = math.hypot(x - rx, y - ry)
            if d < self._min_step_m or d > self._max_step_m:
                continue
            if self._is_near_any((x, y), self._visited, relaxed_revisit):
                continue
            if self._is_near_any((x, y), self._failed, self._failed_blacklist_radius_m):
                continue
            if not self._is_traversable(x, y):
                continue
            return (x, y)

        return None

    def _is_interior_enough(
        self, x: float, y: float, bounds: Tuple[float, float, float, float]
    ) -> bool:
        if len(self._visited) >= self._interior_bias_count:
            return True
        min_x, max_x, min_y, max_y = bounds
        dist_to_edge = min(x - min_x, max_x - x, y - min_y, max_y - y)
        return dist_to_edge >= self._interior_margin_m

    def _is_traversable(self, wx: float, wy: float) -> bool:
        if self._planning_data is None or self._meta is None:
            return True
        idx = self._world_to_grid(wx, wy, self._meta)
        if idx is None:
            return False
        gx, gy = idx
        flat = gx + gy * self._meta.width
        v = self._planning_data[flat]
        if v < 0:
            return True
        return v < self._occ_lethal

    @staticmethod
    def _is_near_any(
        xy: Tuple[float, float], points: List[Tuple[float, float]], radius_m: float
    ) -> bool:
        if radius_m <= 0.0:
            return False
        x, y = xy
        for px, py in points:
            if math.hypot(x - px, y - py) < radius_m:
                return True
        return False

    @staticmethod
    def _point_in_polygon(x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
        inside = False
        n = len(polygon)
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]
            intersects = ((y1 > y) != (y2 > y))
            if not intersects:
                continue
            denom = (y2 - y1)
            if abs(denom) < 1e-9:
                continue
            x_cross = x1 + (y - y1) * (x2 - x1) / denom
            if x_cross > x:
                inside = not inside
        return inside

    @staticmethod
    def _world_to_grid(x: float, y: float, meta: GridMeta) -> Optional[Tuple[int, int]]:
        gx = int(math.floor((x - meta.origin_x) / meta.resolution))
        gy = int(math.floor((y - meta.origin_y) / meta.resolution))
        if gx < 0 or gy < 0 or gx >= meta.width or gy >= meta.height:
            return None
        return (gx, gy)
