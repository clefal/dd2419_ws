#!/usr/bin/env python3

import math
import random
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

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
        min_step_m: float = 1.5,
        max_step_m: float = 2.5,
        min_revisit_dist_m: float = 0.8,
        failed_blacklist_radius_m: float = 0.6,
        exploration_grid_margin_m: float = 0.2,
        occ_lethal: int = 90,
        interior_bias_count: int = 2,
        interior_margin_m: float = 0.4,
        max_samples: int = 500,
        seed: Optional[int] = None,
        logger: Optional[Any] = None,
    ) -> None:
        self._min_step_m = float(min_step_m)
        self._max_step_m = float(max_step_m)
        self._min_revisit_dist_m = float(min_revisit_dist_m)
        self._failed_blacklist_radius_m = float(failed_blacklist_radius_m)
        self._exploration_grid_margin_m = float(exploration_grid_margin_m)
        self._occ_lethal = int(occ_lethal)
        self._interior_bias_count = int(interior_bias_count)
        self._interior_margin_m = float(interior_margin_m)
        self._max_samples = int(max_samples)
        self._rng = random.Random(seed)
        self._logger = logger

        self._workspace_polygon: List[Tuple[float, float]] = []
        self._visited: List[Tuple[float, float]] = []
        self._failed: List[Tuple[float, float]] = []

        self._planning_data: Optional[List[int]] = None
        self._meta: Optional[GridMeta] = None
        self._exploration_data: Optional[List[int]] = None
        self._exploration_meta: Optional[GridMeta] = None

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

    def set_exploration_grid(self, msg: OccupancyGrid) -> None:
        self._exploration_data = list(msg.data)
        self._exploration_meta = GridMeta(
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
        grid_waypoint = self._next_exploration_grid_waypoint(robot_xy)
        if grid_waypoint is not None:
            self._log_info(
                "Explorer selected exploration-grid waypoint: "
                f"({grid_waypoint[0]:.2f},{grid_waypoint[1]:.2f})"
            )
            return grid_waypoint

        if len(self._workspace_polygon) < 3:
            self._log_warn("Explorer has no workspace polygon for random fallback sampling.")
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

            self._log_info(f"Explorer selected random fallback waypoint: ({x:.2f},{y:.2f})")
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
            self._log_info(f"Explorer selected relaxed random fallback waypoint: ({x:.2f},{y:.2f})")
            return (x, y)

        self._log_warn("Explorer failed to find any valid waypoint.")
        return None

    def _next_exploration_grid_waypoint(
        self, robot_xy: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        if self._exploration_data is None or self._exploration_meta is None:
            return None

        waypoint = self._best_unknown_cell_waypoint(
            robot_xy,
            revisit_radius_m=self._min_revisit_dist_m,
        )
        if waypoint is not None:
            return waypoint

        return self._best_unknown_cell_waypoint(
            robot_xy,
            revisit_radius_m=0.35 * self._min_revisit_dist_m,
        )

    def _best_unknown_cell_waypoint(
        self,
        robot_xy: Tuple[float, float],
        revisit_radius_m: float,
    ) -> Optional[Tuple[float, float]]:
        if self._exploration_data is None or self._exploration_meta is None:
            return None

        rx, ry = robot_xy
        meta = self._exploration_meta
        best_xy = None
        best_score = None

        for gy in range(meta.height):
            for gx in range(meta.width):
                value = self._exploration_data[gx + gy * meta.width]
                if not self._is_frontier_cell(gx, gy, meta):
                    continue

                if self._is_too_close_to_exploration_grid_edge(gx, gy, meta):
                    continue

                x, y = self._grid_to_world(gx, gy, meta)
                d_robot = math.hypot(x - rx, y - ry)
                if d_robot < self._min_step_m or d_robot > self._max_step_m:
                    continue

                if self._is_near_any((x, y), self._visited, revisit_radius_m):
                    continue

                if self._is_near_any((x, y), self._failed, self._failed_blacklist_radius_m):
                    continue

                if not self._is_traversable(x, y):
                    continue

                score = (
                    self._distance_to_nearest_point((x, y), self._visited),
                    -d_robot,
                    self._rng.random(),
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_xy = (x, y)

        return best_xy

    def _distance_to_nearest_known_cell(self, gx: int, gy: int) -> float:
        if self._exploration_data is None or self._exploration_meta is None:
            return 0.0

        meta = self._exploration_meta
        max_radius_cells = int(math.ceil(self._max_step_m / meta.resolution))
        best_cells = None

        for dy in range(-max_radius_cells, max_radius_cells + 1):
            ny = gy + dy
            if ny < 0 or ny >= meta.height:
                continue
            for dx in range(-max_radius_cells, max_radius_cells + 1):
                nx = gx + dx
                if nx < 0 or nx >= meta.width:
                    continue
                value = self._exploration_data[nx + ny * meta.width]
                if not self._is_known_exploration_value(value):
                    continue
                dist_cells = math.hypot(dx, dy)
                if best_cells is None or dist_cells < best_cells:
                    best_cells = dist_cells

        if best_cells is None:
            return max_radius_cells * meta.resolution
        return best_cells * meta.resolution

    def _is_frontier_cell(self, gx: int, gy: int, meta: GridMeta) -> bool:
        if self._exploration_data is None:
            return False

        value = self._exploration_data[gx + gy * meta.width]
        if not self._is_unknown_exploration_value(value):
            return False

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx = gx + dx
                ny = gy + dy
                if nx < 0 or ny < 0 or nx >= meta.width or ny >= meta.height:
                    continue
                neighbor_value = self._exploration_data[nx + ny * meta.width]
                if self._is_known_exploration_value(neighbor_value):
                    return True

        return False

    def _is_too_close_to_exploration_grid_edge(self, gx: int, gy: int, meta: GridMeta) -> bool:
        margin_cells = int(math.ceil(self._exploration_grid_margin_m / meta.resolution))
        return (
            gx < margin_cells
            or gy < margin_cells
            or gx >= meta.width - margin_cells
            or gy >= meta.height - margin_cells
        )

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

    def _log_info(self, msg: str) -> None:
        if self._logger is not None:
            self._logger.info(msg)

    def _log_warn(self, msg: str) -> None:
        if self._logger is not None:
            self._logger.warn(msg)

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
    def _distance_to_nearest_point(
        xy: Tuple[float, float], points: List[Tuple[float, float]]
    ) -> float:
        if len(points) == 0:
            return float("inf")
        x, y = xy
        return min(math.hypot(x - px, y - py) for px, py in points)

    @staticmethod
    def _is_unknown_exploration_value(value: int) -> bool:
        return value < 0 or 45 <= value <= 55

    @staticmethod
    def _is_known_exploration_value(value: int) -> bool:
        return 0 <= value < 45

    @staticmethod
    def _grid_to_world(gx: int, gy: int, meta: GridMeta) -> Tuple[float, float]:
        x = meta.origin_x + (float(gx) + 0.5) * meta.resolution
        y = meta.origin_y + (float(gy) + 0.5) * meta.resolution
        return (x, y)

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
