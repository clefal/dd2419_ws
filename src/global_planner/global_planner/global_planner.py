#!/usr/bin/env python3
import math
import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, PoseArray
from tf2_ros import Buffer, TransformListener
from tf_transformations import euler_from_quaternion, quaternion_from_euler


GridIndex = Tuple[int, int]  # (gx, gy)


@dataclass(frozen=True)
class GridMeta:
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float  #  assume ~0 


class GlobalPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__("global_planner")

        # Topics / frames
        self.declare_parameter("map_topic", "/map/occupancy_grid")
        self.declare_parameter("goal_topic", "/nav/goal")
        self.declare_parameter("goal_candidates_topic", "/nav/goal_candidates")
        self.declare_parameter("path_topic", "/nav/global_path")


        # Planning knobs
        self.declare_parameter("w_heuristic", 1.8)          # Weighted A*: f = g + w*h, w=1: normal, w>1: more greedy
        self.declare_parameter("occ_lethal", 90)            # >= lethal => not traversable (0..100) default: 70
        self.declare_parameter("unknown_is_lethal", False)   # OccupancyGrid unknown is -1
        self.declare_parameter("occ_cost_scale", 2.0)       # penalty factor for soft costs
        self.declare_parameter("allow_diagonal", True)
        self.declare_parameter("max_planning_time_ms", 150) # soft guard for very large maps
        self.declare_parameter("cube_approach_radius", 0.16)

        self.map_topic = self.get_parameter("map_topic").get_parameter_value().string_value
        self.goal_topic = self.get_parameter("goal_topic").get_parameter_value().string_value
        self.goal_candidates_topic = self.get_parameter("goal_candidates_topic").get_parameter_value().string_value
        self.path_topic = self.get_parameter("path_topic").get_parameter_value().string_value
        
        self.global_frame = "map"
        self.robot_frame = "base_link"
   
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.sub_map = self.create_subscription(OccupancyGrid, self.map_topic, self.on_map, map_qos)
        self.sub_goal = self.create_subscription(PoseStamped, self.goal_topic, self.on_goal, 10)
        self.sub_goal_candidates = self.create_subscription(
            PoseArray, self.goal_candidates_topic, self.on_goal_candidates, 10
        )
        self.sub_cubes = self.create_subscription(
            PoseArray,
            "/nav/objects/cubes",
            self.on_cubes,
            10,
        )

        self.sub_target_cube = self.create_subscription(
            PoseStamped,
            "/nav/target/cube",
            self.on_target_cube,
            10,
        )


        path_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.pub_path = self.create_publisher(Path, self.path_topic, path_qos)


        planning_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.pub_planning_grid = self.create_publisher(
            OccupancyGrid,
            "/nav/planning_grid",
            planning_qos,
        )



        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        # State
        self._map: Optional[OccupancyGrid] = None
        self._meta: Optional[GridMeta] = None
        self._goal_msg: Optional[PoseStamped] = None
        self._cubes: List[Tuple[float, float]] = []   # in map frame
        self._target_cube: Optional[Tuple[float, float]] = None


        self.get_logger().info(
            f"GlobalPlannerNode started. Subscribing map='{self.map_topic}', goal='{self.goal_topic}', publishing path='{self.path_topic}'."
        )

    # -------------------------
    # ROS callbacks
    # -------------------------
    def on_map(self, msg: OccupancyGrid) -> None:
        self._map = msg
        self._meta = self._extract_meta(msg)

    def on_goal(self, msg: PoseStamped) -> None:
        self._goal_msg = msg
        q = msg.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.get_logger().info(
            f"Goal received: frame='{msg.header.frame_id}', pos=({msg.pose.position.x:.2f},{msg.pose.position.y:.2f}), yaw={yaw:.2f} rad"
        )
        self._plan_and_publish(reason="new_goal")

    def on_goal_candidates(self, msg: PoseArray) -> None:
        self._plan_and_publish_candidates(msg, reason="goal_candidates")


    def on_cubes(self, msg: PoseArray) -> None:
        self._cubes = [
            (p.position.x, p.position.y)
            for p in msg.poses
        ]

    def on_target_cube(self, msg: PoseStamped) -> None:
        self._target_cube = (
            msg.pose.position.x,
            msg.pose.position.y,
        )

    # -------------------------
    # Planning orchestration
    # -------------------------
    def _plan_and_publish(self, reason: str) -> None:
        if self._map is None or self._meta is None:
            self.get_logger().warn("No map yet; cannot plan.")
            return
        if self._goal_msg is None:
            self.get_logger().warn("No goal yet; cannot plan.")
            return

        start_xy = self._get_robot_xy_in_map()
        if start_xy is None:
            self.get_logger().warn("TF unavailable (map->base_link); cannot plan.")
            return

        goal_xy = (self._goal_msg.pose.position.x, self._goal_msg.pose.position.y)

        start_idx = self.world_to_grid(start_xy[0], start_xy[1], self._meta)
        goal_idx = self.world_to_grid(goal_xy[0], goal_xy[1], self._meta)
        self.get_logger().info(
            f"Planning inputs: start_xy=({start_xy[0]:.2f},{start_xy[1]:.2f}) -> {start_idx}, "
            f"goal_xy=({goal_xy[0]:.2f},{goal_xy[1]:.2f}) -> {goal_idx}"
        )

        if start_idx is None or goal_idx is None:
            self.get_logger().warn("Start or goal is outside the grid bounds; cannot plan.")
            self._publish_empty_path(reason="start_or_goal_outside_grid")
            return

        # Run Weighted A*
        planning_map = self.build_planning_grid(self._map, self._meta)
        self.pub_planning_grid.publish(planning_map)

        path_idx = self.weighted_a_star(start_idx, goal_idx, planning_map, self._meta)


        if path_idx is None or len(path_idx) == 0:
            self.get_logger().warn(f"Planning failed ({reason}). No path found.")
            self._publish_empty_path(reason=f"planning_failed_{reason}")
            return

        # For cube goals, cut path at approach radius and face cube
        final_yaw_override = None
        maybe_path, maybe_yaw = self._apply_cube_approach_if_needed(path_idx)
        if maybe_path is not None and len(maybe_path) > 0:
            path_idx = maybe_path
            final_yaw_override = maybe_yaw

        # Convert to nav_msgs/Path in map frame
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self.global_frame

        for (gx, gy) in path_idx:
            wx, wy = self.grid_to_world_center(gx, gy, self._meta)
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = wx
            ps.pose.position.y = wy
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)

        # Set final pose orientation
        if len(path_msg.poses) > 0 and final_yaw_override is not None:
            q = quaternion_from_euler(0.0, 0.0, final_yaw_override)
            path_msg.poses[-1].pose.orientation.x = q[0]
            path_msg.poses[-1].pose.orientation.y = q[1]
            path_msg.poses[-1].pose.orientation.z = q[2]
            path_msg.poses[-1].pose.orientation.w = q[3]
        elif len(path_msg.poses) > 0 and self._goal_msg is not None:
            if self._goal_msg.header.frame_id == self.global_frame or self._goal_msg.header.frame_id == "":
                path_msg.poses[-1].pose.orientation = self._goal_msg.pose.orientation
            else:
                self.get_logger().warn(
                    f"Goal frame '{self._goal_msg.header.frame_id}' != global_frame '{self.global_frame}'. "
                    "Leaving path end orientation as identity."
                )

        self.pub_path.publish(path_msg)
        self.get_logger().info(f"Published path with {len(path_msg.poses)} poses (reason={reason}).")

    def _plan_and_publish_candidates(self, msg: PoseArray, reason: str) -> None:
        if self._map is None or self._meta is None:
            self.get_logger().warn("No map yet; cannot plan candidate goals.")
            return
        if len(msg.poses) == 0:
            self.get_logger().warn("Received empty goal candidate list.")
            self._publish_empty_path(reason="empty_goal_candidates")
            return

        frame = (msg.header.frame_id or "").strip()
        if frame not in ("", self.global_frame):
            self.get_logger().warn(
                f"Goal candidates frame '{frame}' != global_frame '{self.global_frame}'. Ignoring."
            )
            self._publish_empty_path(reason="goal_candidates_wrong_frame")
            return

        start_xy = self._get_robot_xy_in_map()
        if start_xy is None:
            self.get_logger().warn("TF unavailable (map->base_link); cannot plan candidate goals.")
            return

        start_idx = self.world_to_grid(start_xy[0], start_xy[1], self._meta)
        if start_idx is None:
            self.get_logger().warn("Start is outside the grid bounds; cannot plan candidate goals.")
            self._publish_empty_path(reason="start_outside_grid_candidates")
            return

        planning_map = self.build_planning_grid(self._map, self._meta)
        self.pub_planning_grid.publish(planning_map)

        best_path_idx = None
        best_pose = None
        best_cost = None

        for pose in msg.poses:
            goal_idx = self.world_to_grid(pose.position.x, pose.position.y, self._meta)
            if goal_idx is None:
                continue

            path_idx = self.weighted_a_star(start_idx, goal_idx, planning_map, self._meta)
            if path_idx is None or len(path_idx) == 0:
                continue

            pcost = self._path_total_cost(path_idx, planning_map, self._meta)
            if best_cost is None or pcost < best_cost:
                best_cost = pcost
                best_path_idx = path_idx
                best_pose = pose

        if best_path_idx is None or best_pose is None:
            self.get_logger().warn(f"Planning failed ({reason}). No feasible candidate path.")
            self._publish_empty_path(reason=f"planning_failed_{reason}")
            return

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self.global_frame

        for (gx, gy) in best_path_idx:
            wx, wy = self.grid_to_world_center(gx, gy, self._meta)
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = wx
            ps.pose.position.y = wy
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)

        path_msg.poses[-1].pose.orientation = best_pose.orientation
        self.pub_path.publish(path_msg)
        self.get_logger().info(
            f"Published candidate path with {len(path_msg.poses)} poses (reason={reason}, candidates={len(msg.poses)}, cost={best_cost:.2f})."
        )

    def _path_total_cost(self, path_idx: List[GridIndex], occ: OccupancyGrid, meta: GridMeta) -> float:
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

    def _apply_cube_approach_if_needed(
        self, path_idx: List[GridIndex]
    ) -> Tuple[Optional[List[GridIndex]], Optional[float]]:
        if self._goal_msg is None or self._meta is None:
            return (None, None)

        tx = self._goal_msg.pose.position.x
        ty = self._goal_msg.pose.position.y

        radius = self.get_parameter("cube_approach_radius").get_parameter_value().double_value
        if radius <= 0.0:
            return (None, None)

        cut_idx = self._path_index_at_radius(path_idx, tx, ty, radius, self._meta)
        if cut_idx is None:
            return (None, None)

        truncated = path_idx[:cut_idx + 1]
        ax, ay = self.grid_to_world_center(truncated[-1][0], truncated[-1][1], self._meta)
        yaw = math.atan2(ty - ay, tx - ax)
        self.get_logger().info(
            f"Cube approach applied: radius={radius:.2f}, cut_idx={cut_idx}, path_len={len(path_idx)}->{len(truncated)}"
        )
        return (truncated, yaw)

    @staticmethod
    def _path_index_at_radius(
        path_idx: List[GridIndex], tx: float, ty: float, radius: float, meta: GridMeta
    ) -> Optional[int]:
        if len(path_idx) == 0:
            return None

        prev_dist = None
        for i, (gx, gy) in enumerate(path_idx):
            wx, wy = GlobalPlannerNode.grid_to_world_center(gx, gy, meta)
            d = math.hypot(wx - tx, wy - ty)
            if d <= radius:
                if i == 0:
                    return 0
                if prev_dist is None:
                    return i
                return i - 1 if prev_dist > radius else i
            prev_dist = d

        return None

    def _publish_empty_path(self, reason: str = "unknown") -> None:
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self.global_frame
        self.pub_path.publish(path_msg)
        self.get_logger().warn(f"Published EMPTY path (reason={reason}).")

    def _get_robot_xy_in_map(self) -> Optional[Tuple[float, float]]:
        try:
            # latest available transform
            tf = self.tf_buffer.lookup_transform(self.global_frame, self.robot_frame, rclpy.time.Time())
            return (tf.transform.translation.x, tf.transform.translation.y)
        except Exception:
            return None


    def build_planning_grid(self, raw: OccupancyGrid, meta: GridMeta) -> OccupancyGrid:
        # Copy raw map
        planning = OccupancyGrid()
        planning.header = raw.header
        planning.info = raw.info
        lethal = self.get_parameter("occ_lethal").get_parameter_value().integer_value

        robot_radius = 0.05 #0.15
        margin = 0.01
        r_lethal_cells = int(math.ceil((robot_radius + margin) / meta.resolution))

        # soft halo thickness outside the hard core
        soft_halo_m = 0.10
        r_soft_cells = r_lethal_cells + int(math.ceil(soft_halo_m / meta.resolution))

        planning.data = self.inflate_static_obstacles(
            raw.data,
            meta,
            r_lethal_cells,
            r_soft_cells,
            lethal,
)



        cube_radius = 0.015
        r_cells = int(math.ceil(cube_radius / meta.resolution))

        for (cx, cy) in self._cubes:
            if self._target_cube is not None:
                tx, ty = self._target_cube
                if math.hypot(cx - tx, cy - ty) < 0.10:
                    continue  # exclude target cube

            idx = self.world_to_grid(cx, cy, meta)
            if idx is None:
                continue

            self.mark_disk_lethal(planning.data, idx[0], idx[1], r_cells, meta)

        return planning


    def inflate_static_obstacles(self, data, meta, r_lethal, r_soft, lethal_thresh):
        """
        Hard constraint: within r_lethal -> 100 (lethal)
        Soft halo: r_lethal < dist <= r_soft -> descending cost
        """
        inflated = list(data)

        for gy in range(meta.height):
            for gx in range(meta.width):
                v = data[gx + gy * meta.width]

                if v < 0:
                    continue

                if v >= lethal_thresh:
                    for dy in range(-r_soft, r_soft + 1):
                        for dx in range(-r_soft, r_soft + 1):
                            dist2 = dx * dx + dy * dy
                            if dist2 > r_soft * r_soft:
                                continue

                            nx = gx + dx
                            ny = gy + dy
                            if not (0 <= nx < meta.width and 0 <= ny < meta.height):
                                continue

                            if dist2 <= r_lethal * r_lethal:
                                # Hard inflated core
                                inflated[nx + ny * meta.width] = 100
                            else:
                                # Soft cost halo (declines to 0 at r_soft)
                                d = math.sqrt(dist2)
                                t = (d - r_lethal) / max(1e-6, (r_soft - r_lethal))  # 0..1
                                penalty = int(99 * (1.0 - t))  # 99..0
                                idx = nx + ny * meta.width
                                if penalty > inflated[idx]:
                                    inflated[idx] = penalty

        return inflated



    def mark_disk_lethal(self, data, cx, cy, r, meta):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy > r * r:
                    continue

                gx = cx + dx
                gy = cy + dy

                if 0 <= gx < meta.width and 0 <= gy < meta.height:
                    data[gx + gy * meta.width] = 100

    # -------------------------
    # Occupancy grid helpers
    # -------------------------



    @staticmethod
    def _extract_meta(msg: OccupancyGrid) -> GridMeta:
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
        # If origin yaw is non-zero, you should rotate (x,y) into grid frame.
        # Your current mapping publishes yaw ~ 0, so we keep it simple.
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
        # OccupancyGrid data is row-major: index = x + y*width
        return gx + gy * meta.width

    def cell_is_traversable(self, gx: int, gy: int, occ: OccupancyGrid, meta: GridMeta) -> bool:
        lethal = self.get_parameter("occ_lethal").get_parameter_value().integer_value
        unknown_is_lethal = self.get_parameter("unknown_is_lethal").get_parameter_value().bool_value

        v = occ.data[self.idx_to_flat(gx, gy, meta)]  # -1 unknown, 0..100
        if v < 0:
            return not unknown_is_lethal
        return v < lethal

    def cell_penalty(self, gx: int, gy: int, occ: OccupancyGrid, meta: GridMeta) -> float:
        """
        Soft cost for A* to prefer lower occupancy probability.
        Returns >= 0.0
        """
        scale = self.get_parameter("occ_cost_scale").get_parameter_value().double_value
        v = occ.data[self.idx_to_flat(gx, gy, meta)]
        if v < 0:
            # Unknown: treat as moderate penalty if not lethal
            return 0.5 * scale
        # Map 0..100 to 0..scale
        return (float(v) / 100.0) * scale

    # -------------------------
    # Weighted A* implementation
    # -------------------------
    def weighted_a_star(
        self,
        start: GridIndex,
        goal: GridIndex,
        occ: OccupancyGrid,
        meta: GridMeta
    ) -> Optional[List[GridIndex]]:
        w = self.get_parameter("w_heuristic").get_parameter_value().double_value
        allow_diag = self.get_parameter("allow_diagonal").get_parameter_value().bool_value
        max_ms = self.get_parameter("max_planning_time_ms").get_parameter_value().integer_value

        if not self.cell_is_traversable(start[0], start[1], occ, meta):
            v = occ.data[self.idx_to_flat(start[0], start[1], meta)]
            self.get_logger().warn(f"Start cell is not traversable: idx={start}, occ={v}")
            return None
        if not self.cell_is_traversable(goal[0], goal[1], occ, meta):
            v = occ.data[self.idx_to_flat(goal[0], goal[1], meta)]
            self.get_logger().warn(f"Goal cell is not traversable: idx={goal}, occ={v}")
            return None

        # Neighbor moves: (dx, dy, base_cost)
        moves_4 = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0)]
        moves_8 = moves_4 + [(1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))]
        moves = moves_8 if allow_diag else moves_4

        def heuristic(a: GridIndex, b: GridIndex) -> float:
            # Octile distance works well for 8-connected grids
            dx = abs(a[0] - b[0])
            dy = abs(a[1] - b[1])
            if allow_diag:
                return (max(dx, dy) + (math.sqrt(2) - 1.0) * min(dx, dy))
            return float(dx + dy)

        # Priority queue: (f, g, node)
        open_heap: List[Tuple[float, float, GridIndex]] = []
        heapq.heappush(open_heap, (0.0, 0.0, start))

        came_from: Dict[GridIndex, GridIndex] = {}
        g_score: Dict[GridIndex, float] = {start: 0.0}

        start_time = self.get_clock().now()

        while open_heap:
            # soft time guard
            elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e6
            if elapsed > float(max_ms):
                self.get_logger().warn(
                    f"Planning exceeded {max_ms} ms; aborting. expanded={len(g_score)}, open_set={len(open_heap)}"
                )
                return None

            _, g_curr, current = heapq.heappop(open_heap)

            if current == goal:
                return self._reconstruct_path(came_from, current)

            # If this popped entry is stale, skip
            if g_curr > g_score.get(current, float("inf")):
                continue

            cx, cy = current
            for dx, dy, step_cost in moves:
                nx, ny = cx + dx, cy + dy

                if nx < 0 or ny < 0 or nx >= meta.width or ny >= meta.height:
                    continue
                if not self.cell_is_traversable(nx, ny, occ, meta):
                    continue

                # Optional: prevent diagonal "corner cutting"
                if allow_diag and dx != 0 and dy != 0:
                    if not (self.cell_is_traversable(cx + dx, cy, occ, meta) and self.cell_is_traversable(cx, cy + dy, occ, meta)):
                        continue

                penalty = self.cell_penalty(nx, ny, occ, meta)
                tentative_g = g_score[current] + step_cost + penalty

                neighbor = (nx, ny)
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + w * heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (f, tentative_g, neighbor))

        self.get_logger().warn(
            f"Weighted A* exhausted search space without reaching goal. expanded={len(g_score)}"
        )
        return None

    @staticmethod
    def _reconstruct_path(came_from: Dict[GridIndex, GridIndex], current: GridIndex) -> List[GridIndex]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


def main() -> None:
    rclpy.init()
    node = GlobalPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
