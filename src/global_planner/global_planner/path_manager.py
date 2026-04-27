#!/usr/bin/env python3
import heapq
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from nav_msgs.msg import OccupancyGrid
from tf_transformations import euler_from_quaternion

GridIndex = Tuple[int, int]  # (gx, gy)


@dataclass(frozen=True)
class GridMeta:
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float  # assume ~0


@dataclass(frozen=True)
class PlannerConfig:
    w_heuristic: float
    occ_lethal: int
    occ_cost_scale: float
    max_planning_time_ms: int
    path_smoothing_enabled: bool
    path_smoothing_max_shortcut_m: float
    workspace_border_width: float
    robot_radius: float
    inflation_margin: float
    soft_halo_m: float
    cube_size: float
    box_size: float


@dataclass(frozen=True)
class PlanResult:
    path_idx: List[GridIndex]
    planning_grid: OccupancyGrid
    start_idx: GridIndex
    goal_idx: GridIndex


@dataclass(frozen=True)
class CandidatePlanResult:
    path_idx: List[GridIndex]
    planning_grid: OccupancyGrid
    start_idx: GridIndex
    goal_idx: GridIndex
    best_pose_index: int
    cost: float


class PathManager:
    def __init__(self, config: PlannerConfig, logger=None) -> None:
        self._config = config
        self._logger = logger
        self._raw_map: Optional[OccupancyGrid] = None
        self._meta: Optional[GridMeta] = None
        self._planning_grid: Optional[OccupancyGrid] = None
        self._last_include_box_lethal = True
        self._cubes: List[Tuple[float, float]] = []
        self._boxes: List[Tuple[float, float]] = []
        self._target_object: Optional[Tuple[float, float]] = None
        self._workspace_polygon: List[Tuple[float, float]] = []
        self._workspace_mask_meta_key = None
        self._workspace_inside_mask: Optional[List[bool]] = None
        self._workspace_border_mask: Optional[List[bool]] = None

    @property
    def raw_map(self) -> Optional[OccupancyGrid]:
        return self._raw_map

    @property
    def meta(self) -> Optional[GridMeta]:
        return self._meta

    @property
    def planning_grid(self) -> Optional[OccupancyGrid]:
        return self._planning_grid

    def set_config(self, config: PlannerConfig) -> None:
        if config.workspace_border_width != self._config.workspace_border_width:
            self._workspace_mask_meta_key = None
            self._workspace_inside_mask = None
            self._workspace_border_mask = None
        self._config = config

    def update_map(self, msg: OccupancyGrid, rebuild: bool = True) -> None:
        self._raw_map = msg
        self._meta = self.extract_meta(msg)
        if rebuild:
            self.rebuild_planning_grid(include_box_lethal=self._last_include_box_lethal)

    def update_objects(
        self,
        cubes: List[Tuple[float, float]],
        boxes: List[Tuple[float, float]],
        target_object: Optional[Tuple[float, float]],
    ) -> None:
        self._cubes = list(cubes)
        self._boxes = list(boxes)
        self._target_object = target_object

    def set_workspace_polygon(self, polygon_xy: List[Tuple[float, float]]) -> None:
        self._workspace_polygon = list(polygon_xy)
        self._workspace_mask_meta_key = None
        self._workspace_inside_mask = None
        self._workspace_border_mask = None
        if self._raw_map is not None and self._meta is not None:
            self.rebuild_planning_grid(include_box_lethal=self._last_include_box_lethal)

    def rebuild_planning_grid(self, include_box_lethal: bool = True) -> Optional[OccupancyGrid]:
        start_time = time.perf_counter()
        self._last_include_box_lethal = include_box_lethal
        if self._raw_map is None or self._meta is None:
            self._planning_grid = None
            return None
        self._planning_grid = self.build_planning_grid(
            self._raw_map,
            self._meta,
            include_box_lethal=include_box_lethal,
        )
        delta = (time.perf_counter()-start_time)*1000
        self._logger.info(f'replanning took {delta} ms')
        return self._planning_grid

    def plan_to_goal(
        self,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
        include_box_lethal: bool,
        rebuild_grid: bool = True,
    ) -> Optional[PlanResult]:
        if self._raw_map is None or self._meta is None:
            return None

        start_idx = self.world_to_grid(start_xy[0], start_xy[1], self._meta)
        goal_idx = self.world_to_grid(goal_xy[0], goal_xy[1], self._meta)
        if start_idx is None or goal_idx is None:
            return None

        if rebuild_grid:
            planning_grid = self.rebuild_planning_grid(include_box_lethal=include_box_lethal)
        else:
            planning_grid = self._planning_grid
        if planning_grid is None:
            return None

        path_idx = self.weighted_a_star(start_idx, goal_idx, planning_grid, self._meta)
        if path_idx is None or len(path_idx) == 0:
            return None
        path_idx = self.smooth_path(path_idx, planning_grid, self._meta)

        return PlanResult(
            path_idx=path_idx,
            planning_grid=planning_grid,
            start_idx=start_idx,
            goal_idx=goal_idx,
        )

    def plan_to_best_candidate(
        self,
        start_xy: Tuple[float, float],
        candidate_xy: List[Tuple[float, float]],
        include_box_lethal: bool,
        rebuild_grid: bool = True,
    ) -> Optional[CandidatePlanResult]:
        if self._raw_map is None or self._meta is None:
            return None

        start_idx = self.world_to_grid(start_xy[0], start_xy[1], self._meta)
        if start_idx is None:
            return None

        if rebuild_grid:
            planning_grid = self.rebuild_planning_grid(include_box_lethal=include_box_lethal)
        else:
            planning_grid = self._planning_grid
        if planning_grid is None:
            return None

        best_path_idx = None
        best_goal_idx = None
        best_pose_index = None
        best_cost = None

        for pose_index, (goal_x, goal_y) in enumerate(candidate_xy):
            goal_idx = self.world_to_grid(goal_x, goal_y, self._meta)
            if goal_idx is None:
                continue

            path_idx = self.weighted_a_star(start_idx, goal_idx, planning_grid, self._meta)
            if path_idx is None or len(path_idx) == 0:
                continue

            pcost = self.path_total_cost(path_idx, planning_grid, self._meta)
            path_idx = self.smooth_path(path_idx, planning_grid, self._meta)
            if best_cost is None or pcost < best_cost:
                best_cost = pcost
                best_path_idx = path_idx
                best_goal_idx = goal_idx
                best_pose_index = pose_index

        if (
            best_path_idx is None
            or best_goal_idx is None
            or best_pose_index is None
            or best_cost is None
        ):
            return None

        return CandidatePlanResult(
            path_idx=best_path_idx,
            planning_grid=planning_grid,
            start_idx=start_idx,
            goal_idx=best_goal_idx,
            best_pose_index=best_pose_index,
            cost=best_cost,
        )

    def path_is_still_valid(
        self,
        path_idx: List[GridIndex],
        robot_xy: Optional[Tuple[float, float]] = None,
    ) -> bool:
        if self._planning_grid is None or self._meta is None or len(path_idx) == 0:
            return False

        start_i = 0
        if robot_xy is not None:
            robot_idx = self.world_to_grid(robot_xy[0], robot_xy[1], self._meta)
            if robot_idx is not None:
                start_i = self._nearest_path_index(path_idx, robot_idx)
                start_i = max(0, start_i - 1)

        previous = path_idx[start_i]
        for current in path_idx[start_i:]:
            gx, gy = current
            if not (0 <= gx < self._meta.width and 0 <= gy < self._meta.height):
                return False
            if not self.cell_is_traversable(gx, gy, self._planning_grid, self._meta):
                return False
            if not self.segment_is_traversable(previous, current, self._planning_grid, self._meta):
                return False
            previous = current

        return True

    def smooth_path(
        self,
        path_idx: List[GridIndex],
        occ: OccupancyGrid,
        meta: GridMeta,
    ) -> List[GridIndex]:
        if not self._config.path_smoothing_enabled or len(path_idx) <= 2:
            return path_idx

        smoothed: List[GridIndex] = [path_idx[0]]
        anchor_i = 0
        max_shortcut_cells = int(
            math.ceil(max(0.0, self._config.path_smoothing_max_shortcut_m) / meta.resolution)
        )

        while anchor_i < len(path_idx) - 1:
            next_i = anchor_i + 1
            furthest_i = len(path_idx) - 1
            if max_shortcut_cells > 0:
                furthest_i = min(furthest_i, anchor_i + max_shortcut_cells)

            for candidate_i in range(furthest_i, anchor_i, -1):
                if self.segment_is_traversable(path_idx[anchor_i], path_idx[candidate_i], occ, meta):
                    next_i = candidate_i
                    break

            smoothed.append(path_idx[next_i])
            anchor_i = next_i

        self._info(f"Path smoothing: {len(path_idx)} -> {len(smoothed)} waypoints")
        return smoothed

    def segment_is_traversable(
        self,
        start: GridIndex,
        goal: GridIndex,
        occ: OccupancyGrid,
        meta: GridMeta,
    ) -> bool:
        x0, y0 = start
        x1, y1 = goal
        dx = x1 - x0
        dy = y1 - y0
        steps = max(abs(dx), abs(dy)) * 2
        if steps <= 0:
            return self.cell_is_traversable(x0, y0, occ, meta)

        previous = None
        for i in range(steps + 1):
            t = float(i) / float(steps)
            gx = int(round(x0 + dx * t))
            gy = int(round(y0 + dy * t))
            current = (gx, gy)
            if current == previous:
                continue
            previous = current

            if not (0 <= gx < meta.width and 0 <= gy < meta.height):
                return False
            if not self.cell_is_traversable(gx, gy, occ, meta):
                return False

        return True



    def build_planning_grid(
        self, raw: OccupancyGrid, meta: GridMeta, include_box_lethal: bool = True
    ) -> OccupancyGrid:
        planning = OccupancyGrid()
        planning.header = raw.header
        planning.info = raw.info

        r_lethal_cells = int(
            math.ceil((self._config.robot_radius + self._config.inflation_margin) / meta.resolution)
        )

        r_soft_cells = r_lethal_cells + int(
            math.ceil(self._config.soft_halo_m / meta.resolution)
        )

        static_data = list(raw.data)
        workspace_inside, workspace_border = self._workspace_masks(meta)
        if workspace_inside is not None:
            # Ignore the filled outside-workspace region as an inflation source.
            for i, is_inside in enumerate(workspace_inside):
                if not is_inside:
                    static_data[i] = 0
        if workspace_border is not None:
            for i, is_border in enumerate(workspace_border):
                if is_border:
                    static_data[i] = 100

        planning.data = self.inflate_static_obstacles(
            static_data,
            meta,
            r_lethal_cells,
            r_soft_cells,
            self._config.occ_lethal,
        )
        if workspace_inside is not None:
            # Keep the published planning grid visually and semantically non-traversable outside.
            for i, is_inside in enumerate(workspace_inside):
                if not is_inside:
                    planning.data[i] = 100

        cube_half_diagonal = 0.5 * self._config.cube_size * math.sqrt(2.0)
        cube_keepout_radius = (
            self._config.robot_radius + cube_half_diagonal + self._config.inflation_margin
        )
        r_cells = int(math.ceil(cube_keepout_radius / meta.resolution))
        r_soft_object_cells = r_cells + int(
            math.ceil(self._config.soft_halo_m / meta.resolution)
        )

        for (cx, cy) in self._cubes:
            if self._target_object is not None:
                tx, ty = self._target_object
                if math.hypot(cx - tx, cy - ty) < 0.10:
                    continue

            idx = self.world_to_grid(cx, cy, meta)
            if idx is None:
                continue

            self.mark_disk_inflated(
                planning.data,
                idx[0],
                idx[1],
                r_cells,
                r_soft_object_cells,
                meta,
            )

        if include_box_lethal:
            box_keepout_radius = (
                self._config.robot_radius
                + self._config.box_size
                + self._config.inflation_margin
            )
            box_r_cells = int(math.ceil(box_keepout_radius / meta.resolution))
            box_r_soft_cells = box_r_cells + int(
                math.ceil(self._config.soft_halo_m / meta.resolution)
            )

            for (bx, by) in self._boxes:
                box_idx = self.world_to_grid(bx, by, meta)
                if box_idx is None:
                    continue

                self.mark_disk_inflated(
                    planning.data,
                    box_idx[0],
                    box_idx[1],
                    box_r_cells,
                    box_r_soft_cells,
                    meta,
                )

        return planning

    def inflate_static_obstacles(self, data, meta, r_lethal, r_soft, lethal_thresh):
        """
        Hard constraint: within r_lethal -> 100 (lethal)
        Soft halo: r_lethal < dist <= r_soft -> descending cost
        """
        inflated = list(data)
        cache_key = (r_lethal, r_soft)
        if not hasattr(self, "_inflation_offsets_cache"):
            self._inflation_offsets_cache = {}

        offsets = self._inflation_offsets_cache.get(cache_key)
        if offsets is None:
            offsets = []
            r_lethal2 = r_lethal * r_lethal
            r_soft2 = r_soft * r_soft
            for dy in range(-r_soft, r_soft + 1):
                for dx in range(-r_soft, r_soft + 1):
                    dist2 = dx * dx + dy * dy
                    if dist2 > r_soft2:
                        continue

                    if dist2 <= r_lethal2:
                        cost = 100
                    else:
                        d = math.sqrt(dist2)
                        t = (d - r_lethal) / max(1e-6, (r_soft - r_lethal))
                        cost = int(99 * (1.0 - t))

                    offsets.append((dx, dy, cost))

            self._inflation_offsets_cache[cache_key] = offsets

        for gy in range(meta.height):
            for gx in range(meta.width):
                v = data[gx + gy * meta.width]

                if v < 0:
                    continue

                if v >= lethal_thresh:
                    for dx, dy, cost in offsets:
                        nx = gx + dx
                        ny = gy + dy
                        if not (0 <= nx < meta.width and 0 <= ny < meta.height):
                            continue

                        idx = nx + ny * meta.width
                        if cost == 100:
                            inflated[idx] = 100
                        elif cost > inflated[idx]:
                            inflated[idx] = cost

        return inflated

    @staticmethod
    def mark_disk_lethal(data, cx, cy, r, meta):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy > r * r:
                    continue

                gx = cx + dx
                gy = cy + dy

                if 0 <= gx < meta.width and 0 <= gy < meta.height:
                    data[gx + gy * meta.width] = 100

    @staticmethod
    def mark_disk_inflated(data, cx, cy, r_lethal, r_soft, meta):
        for dy in range(-r_soft, r_soft + 1):
            for dx in range(-r_soft, r_soft + 1):
                dist2 = dx * dx + dy * dy
                if dist2 > r_soft * r_soft:
                    continue

                gx = cx + dx
                gy = cy + dy
                if not (0 <= gx < meta.width and 0 <= gy < meta.height):
                    continue

                idx = gx + gy * meta.width
                if dist2 <= r_lethal * r_lethal:
                    data[idx] = 100
                else:
                    d = math.sqrt(dist2)
                    t = (d - r_lethal) / max(1e-6, (r_soft - r_lethal))
                    penalty = int(99 * (1.0 - t))
                    if penalty > data[idx]:
                        data[idx] = penalty

    def _workspace_masks(
        self,
        meta: GridMeta,
    ) -> Tuple[Optional[List[bool]], Optional[List[bool]]]:
        if len(self._workspace_polygon) < 3:
            return (None, None)

        meta_key = (
            meta.width,
            meta.height,
            meta.resolution,
            meta.origin_x,
            meta.origin_y,
            self._config.workspace_border_width,
            tuple(self._workspace_polygon),
        )
        if self._workspace_mask_meta_key == meta_key:
            return (self._workspace_inside_mask, self._workspace_border_mask)

        inside_mask = [False] * (meta.width * meta.height)
        border_mask = [False] * (meta.width * meta.height)
        border_width = max(0.0, float(self._config.workspace_border_width))

        for gy in range(meta.height):
            wy = meta.origin_y + (gy + 0.5) * meta.resolution
            for gx in range(meta.width):
                wx = meta.origin_x + (gx + 0.5) * meta.resolution
                idx = gx + gy * meta.width
                inside_mask[idx] = self._point_in_polygon(wx, wy, self._workspace_polygon)
                border_mask[idx] = self._point_near_polygon_edge(
                    wx,
                    wy,
                    self._workspace_polygon,
                    border_width,
                )

        self._workspace_mask_meta_key = meta_key
        self._workspace_inside_mask = inside_mask
        self._workspace_border_mask = border_mask
        return (inside_mask, border_mask)

    @staticmethod
    def _point_on_segment(px, py, ax, ay, bx, by, eps=1e-9) -> bool:
        abx = bx - ax
        aby = by - ay
        apx = px - ax
        apy = py - ay
        cross = abx * apy - aby * apx
        if abs(cross) > eps:
            return False
        dot = apx * abx + apy * aby
        if dot < -eps:
            return False
        sq_len = abx * abx + aby * aby
        if dot - sq_len > eps:
            return False
        return True

    @classmethod
    def _point_in_polygon(cls, px, py, polygon: List[Tuple[float, float]]) -> bool:
        inside = False
        n = len(polygon)
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]

            if cls._point_on_segment(px, py, x1, y1, x2, y2):
                return True

            intersects = ((y1 > py) != (y2 > py)) and (
                px < (x2 - x1) * (py - y1) / ((y2 - y1) if (y2 - y1) != 0.0 else 1e-12) + x1
            )
            if intersects:
                inside = not inside
        return inside

    @staticmethod
    def _point_segment_distance(px, py, ax, ay, bx, by) -> float:
        abx = bx - ax
        aby = by - ay
        apx = px - ax
        apy = py - ay
        sq_len = abx * abx + aby * aby
        if sq_len <= 1e-12:
            return math.hypot(px - ax, py - ay)
        t = (apx * abx + apy * aby) / sq_len
        t = max(0.0, min(1.0, t))
        closest_x = ax + t * abx
        closest_y = ay + t * aby
        return math.hypot(px - closest_x, py - closest_y)

    @classmethod
    def _point_near_polygon_edge(
        cls,
        px,
        py,
        polygon: List[Tuple[float, float]],
        max_dist: float,
    ) -> bool:
        n = len(polygon)
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]
            if cls._point_segment_distance(px, py, x1, y1, x2, y2) <= max_dist:
                return True
        return False

    @staticmethod
    def extract_meta(msg: OccupancyGrid) -> GridMeta:
        ox = msg.info.origin.position.x
        oy = msg.info.origin.position.y
        q = msg.info.origin.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        return GridMeta(
            width=msg.info.width,
            height=msg.info.height,
            resolution=msg.info.resolution,
            origin_x=ox,
            origin_y=oy,
            origin_yaw=yaw,
        )

    @staticmethod
    def world_to_grid(x: float, y: float, meta: GridMeta) -> Optional[GridIndex]:
        gx = int(math.floor((x - meta.origin_x) / meta.resolution))
        gy = int(math.floor((y - meta.origin_y) / meta.resolution))
        if gx < 0 or gy < 0 or gx >= meta.width or gy >= meta.height:
            return None
        return (gx, gy)

    @staticmethod
    def grid_to_world_center(gx: int, gy: int, meta: GridMeta) -> Tuple[float, float]:
        x = meta.origin_x + (gx + 0.5) * meta.resolution
        y = meta.origin_y + (gy + 0.5) * meta.resolution
        return (x, y)

    @staticmethod
    def idx_to_flat(gx: int, gy: int, meta: GridMeta) -> int:
        return gx + gy * meta.width

    def cell_is_traversable(self, gx: int, gy: int, occ: OccupancyGrid, meta: GridMeta) -> bool:
        if len(self._workspace_polygon) >= 3 and self._workspace_inside_mask is None:
            self._workspace_masks(meta)
        if (
            self._workspace_inside_mask is not None
            and not self._workspace_inside_mask[self.idx_to_flat(gx, gy, meta)]
        ):
            return False

        v = occ.data[self.idx_to_flat(gx, gy, meta)]
        if v < 0:
            return True
        return v < self._config.occ_lethal

    def cell_penalty(self, gx: int, gy: int, occ: OccupancyGrid, meta: GridMeta) -> float:
        v = occ.data[self.idx_to_flat(gx, gy, meta)]
        if v < 0:
            return 0.2 * self._config.occ_cost_scale
        return (float(v) / 100.0) * self._config.occ_cost_scale

    def path_total_cost(self, path_idx: List[GridIndex], occ: OccupancyGrid, meta: GridMeta) -> float:
        if len(path_idx) <= 1:
            return 0.0

        total = 0.0
        for i in range(1, len(path_idx)):
            x0, y0 = path_idx[i - 1]
            x1, y1 = path_idx[i]
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            step = math.sqrt(2.0) if (dx == 1 and dy == 1) else 1.0
            total += step + self.cell_penalty(x1, y1, occ, meta)
        return total

    def weighted_a_star(
        self,
        start: GridIndex,
        goal: GridIndex,
        occ: OccupancyGrid,
        meta: GridMeta,
    ) -> Optional[List[GridIndex]]:
        if not self.cell_is_traversable(start[0], start[1], occ, meta):
            v = occ.data[self.idx_to_flat(start[0], start[1], meta)]
            self._warn(f"Start cell is not traversable: idx={start}, occ={v}")
            return None
        if not self.cell_is_traversable(goal[0], goal[1], occ, meta):
            v = occ.data[self.idx_to_flat(goal[0], goal[1], meta)]
            self._warn(f"Goal cell is not traversable: idx={goal}, occ={v}")
            return None

        moves_4 = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0)]
        moves_8 = moves_4 + [
            (1, 1, math.sqrt(2)),
            (1, -1, math.sqrt(2)),
            (-1, 1, math.sqrt(2)),
            (-1, -1, math.sqrt(2)),
        ]

        def heuristic(a: GridIndex, b: GridIndex) -> float:
            dx = abs(a[0] - b[0])
            dy = abs(a[1] - b[1])
            return max(dx, dy) + (math.sqrt(2) - 1.0) * min(dx, dy)

        open_heap: List[Tuple[float, float, GridIndex]] = []
        heapq.heappush(open_heap, (0.0, 0.0, start))

        came_from: Dict[GridIndex, GridIndex] = {}
        g_score: Dict[GridIndex, float] = {start: 0.0}

        start_time = time.monotonic()
        self._info(f"Starting A*: start={start}, goal={goal}")

        while open_heap:
            elapsed_ms = (time.monotonic() - start_time) * 1000.0
            if elapsed_ms > float(self._config.max_planning_time_ms):
                self._info(
                    f"Finished A*: status=timeout, elapsed_ms={elapsed_ms:.1f}, "
                    f"expanded={len(g_score)}, open_set={len(open_heap)}"
                )
                self._warn(
                    f"Planning exceeded {self._config.max_planning_time_ms} ms; "
                    f"aborting. expanded={len(g_score)}, open_set={len(open_heap)}"
                )
                return None

            _, g_curr, current = heapq.heappop(open_heap)

            if current == goal:
                path = self._reconstruct_path(came_from, current)
                elapsed_ms = (time.monotonic() - start_time) * 1000.0
                self._info(
                    f"Finished A*: status=success, elapsed_ms={elapsed_ms:.1f}, "
                    f"expanded={len(g_score)}, open_set={len(open_heap)}, path_len={len(path)}"
                )
                return path

            if g_curr > g_score.get(current, float("inf")):
                continue

            cx, cy = current
            for dx, dy, step_cost in moves_8:
                nx, ny = cx + dx, cy + dy

                if nx < 0 or ny < 0 or nx >= meta.width or ny >= meta.height:
                    continue
                if not self.cell_is_traversable(nx, ny, occ, meta):
                    continue

                if dx != 0 and dy != 0:
                    if not (
                        self.cell_is_traversable(cx + dx, cy, occ, meta)
                        and self.cell_is_traversable(cx, cy + dy, occ, meta)
                    ):
                        continue

                penalty = self.cell_penalty(nx, ny, occ, meta)
                tentative_g = g_score[current] + step_cost + penalty

                neighbor = (nx, ny)
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._config.w_heuristic * heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (f, tentative_g, neighbor))

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        self._info(
            f"Finished A*: status=exhausted, elapsed_ms={elapsed_ms:.1f}, "
            f"expanded={len(g_score)}, open_set={len(open_heap)}"
        )
        self._warn(f"Weighted A* exhausted search space without reaching goal. expanded={len(g_score)}")
        return None

    @staticmethod
    def _reconstruct_path(came_from: Dict[GridIndex, GridIndex], current: GridIndex) -> List[GridIndex]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    @staticmethod
    def _nearest_path_index(path_idx: List[GridIndex], robot_idx: GridIndex) -> int:
        best_i = 0
        best_dist2 = None
        rx, ry = robot_idx
        for i, (gx, gy) in enumerate(path_idx):
            dist2 = (gx - rx) * (gx - rx) + (gy - ry) * (gy - ry)
            if best_dist2 is None or dist2 < best_dist2:
                best_dist2 = dist2
                best_i = i
        return best_i

    def _warn(self, msg: str) -> None:
        if self._logger is not None:
            self._logger.warn(msg)

    def _info(self, msg: str) -> None:
        if self._logger is not None:
            self._logger.info(msg)
