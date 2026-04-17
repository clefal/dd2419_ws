#!/usr/bin/env python3
import math
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, PoseArray, PolygonStamped
from tf2_ros import Buffer, TransformListener
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from robp_interfaces.srv import GetAllObjects
from robp_interfaces.msg import ObjPose
from std_srvs.srv import Trigger

from .path_manager import GridIndex, GridMeta, PathManager, PlannerConfig


class GlobalPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__("global_planner")

        # Topics / frames
        self.declare_parameter("map_topic", "/map/occupancy_grid")
        self.declare_parameter("goal_topic", "/nav/goal")
        self.declare_parameter("goal_candidates_topic", "/nav/goal_candidates")
        self.declare_parameter("path_topic", "/nav/global_path")
        self.declare_parameter("workspace_topic", "/workspace")


        # Planning knobs
        self.declare_parameter("w_heuristic", 1.8)          # Weighted A*: f = g + w*h, w=1: normal, w>1: more greedy
        self.declare_parameter("occ_lethal", 90)            # >= lethal => not traversable (0..100) default: 70
        self.declare_parameter("occ_cost_scale", 2.0)       # penalty factor for soft costs
        self.declare_parameter("max_planning_time_ms", 150) # soft guard for very large maps
        self.declare_parameter("workspace_border_width", 0.05)
        self.declare_parameter("coarse_object_standoff", 0.5)
        self.declare_parameter("robot_radius", 0.05)
        self.declare_parameter("inflation_margin", 0.01)
        self.declare_parameter("soft_halo_m", 0.10)
        self.declare_parameter("cube_size", 0.02)
        self.declare_parameter("box_size", 0.16)
        self.declare_parameter("box_goal_radius", 0.30)
        self.declare_parameter("replan_check_period_s", 0.5)
        self.declare_parameter("freeze_odom_before_planning", False)
        self.declare_parameter("freeze_odom_service", "/localization/freeze_odom")
        self.declare_parameter("freeze_odom_timeout_s", 0.5)

        self.map_topic = self.get_parameter("map_topic").get_parameter_value().string_value
        self.goal_topic = self.get_parameter("goal_topic").get_parameter_value().string_value
        self.goal_candidates_topic = self.get_parameter("goal_candidates_topic").get_parameter_value().string_value
        self.path_topic = self.get_parameter("path_topic").get_parameter_value().string_value
        self.workspace_topic = self.get_parameter("workspace_topic").get_parameter_value().string_value
        
        self.global_frame = "map"
        self.robot_frame = "base_link"
   
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.sub_map = self.create_subscription(OccupancyGrid, self.map_topic, self.on_map, map_qos)
        self.sub_workspace = self.create_subscription(
            PolygonStamped,
            self.workspace_topic,
            self.on_workspace,
            map_qos,
        )
        self.sub_goal = self.create_subscription(PoseStamped, self.goal_topic, self.on_goal, 10) 
        # TODO we could change this to a message type that includes x,y and obj_id of the goal, that could make the list update cleaner
        self.sub_goal_candidates = self.create_subscription(
            PoseArray, self.goal_candidates_topic, self.on_goal_candidates, 10
        )

        self.cli_get_all_objects = self.create_client(GetAllObjects, '/object_manager/get_all_objects')
        while not self.cli_get_all_objects.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('get_all_objects service not available, waiting again...')       


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
            '/nav/planning_grid',
            planning_qos,
        )

        self.path_manager = PathManager(self._planner_config(), logger=self.get_logger())

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        self.freeze_odom_before_planning = self.get_parameter("freeze_odom_before_planning").value
        self.freeze_odom_service = self.get_parameter("freeze_odom_service").value
        self.freeze_odom_timeout_s = float(self.get_parameter("freeze_odom_timeout_s").value)
        self.cli_freeze_odom = None
        if self.freeze_odom_before_planning:
            self.cli_freeze_odom = self.create_client(Trigger, self.freeze_odom_service)

        # State
        self._goal_msg: Optional[PoseStamped] = None
        self._cubes: List[Tuple[float, float]] = []   # in map frame
        self._boxes: List[Tuple[float, float]] = []   # in map frame
        self._target_object: Optional[Tuple[float, float]] = None
        self.goal_x = None  # initialize this with None, value will be assigned during first goal callback
        self.goal_y = None
        self._pending_plan_mode: Optional[str] = None
        self._pending_plan_reason: str = "unknown"
        self._pending_goal_candidates: Optional[PoseArray] = None
        self._active_plan_mode: Optional[str] = None
        self._active_goal_candidates: Optional[PoseArray] = None
        self._current_path_idx: Optional[List[GridIndex]] = None
        self._current_include_box_lethal = True
        self._replan_in_progress = False
        self._replan_check_pending = False

        replan_period = self.get_parameter("replan_check_period_s").get_parameter_value().double_value
        self._replan_timer = self.create_timer(replan_period, self._check_replan)


        self.get_logger().info(
            f"GlobalPlannerNode started. Subscribing map='{self.map_topic}', goal='{self.goal_topic}', publishing path='{self.path_topic}'."
        )

    # -------------------------
    # ROS callbacks
    # -------------------------
    def _planner_config(self) -> PlannerConfig:
        return PlannerConfig(
            w_heuristic=self.get_parameter("w_heuristic").get_parameter_value().double_value,
            occ_lethal=self.get_parameter("occ_lethal").get_parameter_value().integer_value,
            occ_cost_scale=self.get_parameter("occ_cost_scale").get_parameter_value().double_value,
            max_planning_time_ms=self.get_parameter("max_planning_time_ms").get_parameter_value().integer_value,
            workspace_border_width=self.get_parameter("workspace_border_width").get_parameter_value().double_value,
            robot_radius=self.get_parameter("robot_radius").get_parameter_value().double_value,
            inflation_margin=self.get_parameter("inflation_margin").get_parameter_value().double_value,
            soft_halo_m=self.get_parameter("soft_halo_m").get_parameter_value().double_value,
            cube_size=self.get_parameter("cube_size").get_parameter_value().double_value,
            box_size=self.get_parameter("box_size").get_parameter_value().double_value,
            box_goal_radius=self.get_parameter("box_goal_radius").get_parameter_value().double_value,
        )

    def on_map(self, msg: OccupancyGrid) -> None:
        self.path_manager.set_config(self._planner_config())
        self.path_manager.update_map(msg)
        planning_grid = self.path_manager.planning_grid
        if planning_grid is not None:
            self.pub_planning_grid.publish(planning_grid)

    def on_workspace(self, msg: PolygonStamped) -> None:
        frame = (msg.header.frame_id or "").strip()
        if frame not in ("", self.global_frame):
            self.get_logger().warn(
                f"Workspace frame '{frame}' != global_frame '{self.global_frame}'. Ignoring."
            )
            return

        polygon_xy = [(float(p.x), float(p.y)) for p in msg.polygon.points]
        if len(polygon_xy) < 3:
            self.get_logger().warn("Workspace polygon has fewer than 3 points. Ignoring.")
            return

        self.path_manager.set_config(self._planner_config())
        self.path_manager.set_workspace_polygon(polygon_xy)
        planning_grid = self.path_manager.planning_grid
        if planning_grid is not None:
            self.pub_planning_grid.publish(planning_grid)
        self.get_logger().info(f"Workspace polygon loaded for planner ({len(polygon_xy)} points).")

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


    # -------------------------
    # Planning orchestration
    # -------------------------
    def _plan_and_publish(self, reason: str) -> None:
        if self.path_manager.raw_map is None or self.path_manager.meta is None:
            self.get_logger().warn("No map yet; cannot plan.")
            return
        if self._goal_msg is None:
            self.get_logger().warn("No goal yet; cannot plan.")
            return

        # now call the GetAllObj Service and update the List accordingly
        # make sure to exclude the goal position from the Object List, otherwise we will black the goal out
        # updateobject_list needs goal pose in the future callback that is why we need it as a global variable
        goal_xy = (self._goal_msg.pose.position.x, self._goal_msg.pose.position.y) # moved up since it is needed for the Obj_List_update
        self.goal_x = goal_xy[0]
        self.goal_y = goal_xy[1]
        self._pending_plan_mode = "single"
        self._pending_plan_reason = reason
        self._current_path_idx = None

        self.update_object_list()
        return

    def freeze_odom_for_planning(self, done_callback) -> None:
        if not self.freeze_odom_before_planning:
            done_callback(True)
            return

        if self.cli_freeze_odom is None:
            self.get_logger().warn("freeze_odom_before_planning is enabled, but service client is missing.")
            done_callback(False)
            return

        if not self.cli_freeze_odom.wait_for_service(timeout_sec=self.freeze_odom_timeout_s):
            self.get_logger().warn(f"Freeze odom service unavailable: {self.freeze_odom_service}")
            done_callback(False)
            return

        self.get_logger().info(f"Requesting map->odom freeze before planning via {self.freeze_odom_service}")
        future = self.cli_freeze_odom.call_async(Trigger.Request())
        future.add_done_callback(
            lambda freeze_future: self.freeze_odom_done_callback(freeze_future, done_callback)
        )

    def freeze_odom_done_callback(self, future, done_callback) -> None:
        try:
            res = future.result()
        except Exception as ex:
            self.get_logger().warn(f"Freeze odom service call failed: {ex}")
            done_callback(False)
            return

        if not res.success:
            self.get_logger().warn(f"Freeze odom service rejected request: {res.message}")
            done_callback(False)
            return

        self.get_logger().info(res.message)
        done_callback(True)

    def update_object_list(self):
        req = GetAllObjects.Request()
        future = self.cli_get_all_objects.call_async(req)
        future.add_done_callback(self.get_all_objects_callback)

    def get_all_objects_callback(self, future):
        
        try:
            res :GetAllObjects.Response = future.result()
            obj_poses :List[ObjPose] = res.obj_poses
            self._cubes: List[Tuple[float, float]] = []   # in map frame
            self._boxes: List[Tuple[float, float]] = []   # in map frame
            for obj in obj_poses:
                if obj.obj_type == 'box':
                    self._boxes.append((obj.obj_x, obj.obj_y))
                else:
                    self._cubes.append((obj.obj_x, obj.obj_y))

            self.path_manager.set_config(self._planner_config())
            self.path_manager.update_objects(
                cubes=self._cubes,
                boxes=self._boxes,
                target_object=self._target_object,
            )

            if self._replan_check_pending:
                self._continue_replan_check()
            elif self._pending_plan_mode == "single":
                self._continue_plan_and_publish(reason=self._pending_plan_reason)
            elif self._pending_plan_mode == "candidates":
                self._continue_plan_and_publish_candidates(reason=self._pending_plan_reason)

        except Exception as e:
            self._replan_check_pending = False
            self._replan_in_progress = False
            self.get_logger().info(f'get_all_objects service call failed {e}')
        
        
    def _plan_and_publish_candidates(self, msg: PoseArray, reason: str) -> None:
        if self.path_manager.raw_map is None or self.path_manager.meta is None:
            self.get_logger().warn("No map yet; cannot plan candidate goals.")
            return
        if len(msg.poses) == 0:
            self.get_logger().warn("Received empty goal candidate list.")
            self._publish_empty_path(reason="empty_goal_candidates")
            return
    
        goal_box_avg_x = sum(pose.position.x for pose in msg.poses) / len(msg.poses)
        goal_box_avg_y = sum(pose.position.y for pose in msg.poses) / len(msg.poses)
        # Use the candidate centroid for the object-list exclusion bookkeeping.
        
        self.goal_x = goal_box_avg_x
        self.goal_y = goal_box_avg_y
        self._pending_goal_candidates = msg
        self._pending_plan_mode = "candidates"
        self._pending_plan_reason = reason
        self._current_path_idx = None

        self.update_object_list()   
        return

    def _continue_plan_and_publish(self, reason: str) -> None:
        meta = self.path_manager.meta
        if self.path_manager.raw_map is None or meta is None:
            self.get_logger().warn("No map yet; cannot plan.")
            return
        if self._goal_msg is None:
            self.get_logger().warn("No goal yet; cannot plan.")
            return

        self.freeze_odom_for_planning(
            lambda ok: self._continue_plan_and_publish_after_freeze(reason, ok)
        )
        return

    def _continue_plan_and_publish_after_freeze(self, reason: str, freeze_ok: bool) -> None:
        if not freeze_ok:
            self._publish_empty_path(reason="freeze_odom_failed")
            return

        meta = self.path_manager.meta
        if self.path_manager.raw_map is None or meta is None:
            self.get_logger().warn("No map yet; cannot plan.")
            return
        if self._goal_msg is None:
            self.get_logger().warn("No goal yet; cannot plan.")
            return

        goal_xy = (self._goal_msg.pose.position.x, self._goal_msg.pose.position.y)
        self._pending_plan_mode = None
        self._pending_plan_reason = "unknown"
        self._replan_in_progress = False

        start_xy = self._get_robot_xy_in_map()
        if start_xy is None:
            self.get_logger().warn("TF unavailable (map->base_link); cannot plan.")
            return

        start_idx = self.path_manager.world_to_grid(start_xy[0], start_xy[1], meta)
        goal_idx = self.path_manager.world_to_grid(goal_xy[0], goal_xy[1], meta)
        self.get_logger().info(
            f"Planning inputs: start_xy=({start_xy[0]:.2f},{start_xy[1]:.2f}) -> {start_idx}, "
            f"goal_xy=({goal_xy[0]:.2f},{goal_xy[1]:.2f}) -> {goal_idx}"
        )

        if start_idx is None or goal_idx is None:
            self.get_logger().warn("Start or goal is outside the grid bounds; cannot plan.")
            self._publish_empty_path(reason="start_or_goal_outside_grid")
            return

        include_box_lethal = True
        plan = self.path_manager.plan_to_goal(start_xy, goal_xy, include_box_lethal=include_box_lethal)
        planning_grid = self.path_manager.planning_grid
        if planning_grid is not None:
            self.pub_planning_grid.publish(planning_grid)

        if plan is None:
            self.get_logger().warn(f"Planning failed ({reason}). No path found.")
            self._publish_empty_path(reason=f"planning_failed_{reason}")
            return

        path_idx = plan.path_idx
        final_yaw_override = None
        maybe_path, maybe_yaw = self._apply_coarse_object_standoff(path_idx)
        if maybe_path is not None and len(maybe_path) > 0:
            path_idx = maybe_path
            final_yaw_override = maybe_yaw

        goal_orientation = None
        if final_yaw_override is None and self._goal_msg is not None:
            if self._goal_msg.header.frame_id == self.global_frame or self._goal_msg.header.frame_id == "":
                goal_orientation = self._goal_msg.pose.orientation
            else:
                self.get_logger().warn(
                    f"Goal frame '{self._goal_msg.header.frame_id}' != global_frame '{self.global_frame}'. "
                    "Leaving path end orientation as identity."
                )

        path_msg = self._build_path_message(
            path_idx,
            final_yaw=final_yaw_override,
            final_orientation=goal_orientation,
        )
        self.pub_path.publish(path_msg)
        self._current_path_idx = path_idx
        self._current_include_box_lethal = include_box_lethal
        self._active_plan_mode = "single"
        self._active_goal_candidates = None
        self._replan_in_progress = False
        self.get_logger().info(f"Published path with {len(path_msg.poses)} poses (reason={reason}).")

    def _continue_plan_and_publish_candidates(self, reason: str) -> None:
        msg = self._pending_goal_candidates
        self._pending_plan_mode = None
        self._pending_plan_reason = "unknown"
        self._pending_goal_candidates = None
        self._replan_in_progress = False

        meta = self.path_manager.meta
        if self.path_manager.raw_map is None or meta is None:
            self.get_logger().warn("No map yet; cannot plan candidate goals.")
            return
        if msg is None:
            self.get_logger().warn("No goal candidates yet; cannot plan candidate goals.")
            return
        if len(msg.poses) == 0:
            self.get_logger().warn("Received empty goal candidate list.")
            self._publish_empty_path(reason="empty_goal_candidates")
            return

        self.freeze_odom_for_planning(
            lambda ok: self._continue_plan_and_publish_candidates_after_freeze(reason, msg, ok)
        )
        return

    def _continue_plan_and_publish_candidates_after_freeze(
        self,
        reason: str,
        msg: PoseArray,
        freeze_ok: bool,
    ) -> None:
        if not freeze_ok:
            self._publish_empty_path(reason="freeze_odom_failed_candidates")
            return

        meta = self.path_manager.meta
        if self.path_manager.raw_map is None or meta is None:
            self.get_logger().warn("No map yet; cannot plan candidate goals.")
            return
        if msg is None:
            self.get_logger().warn("No goal candidates yet; cannot plan candidate goals.")
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

        start_idx = self.path_manager.world_to_grid(start_xy[0], start_xy[1], meta)
        if start_idx is None:
            self.get_logger().warn("Start is outside the grid bounds; cannot plan candidate goals.")
            self._publish_empty_path(reason="start_outside_grid_candidates")
            return

        include_box_lethal = True
        candidate_xy = [(pose.position.x, pose.position.y) for pose in msg.poses]
        plan = self.path_manager.plan_to_best_candidate(
            start_xy,
            candidate_xy,
            include_box_lethal=include_box_lethal,
        )
        planning_grid = self.path_manager.planning_grid
        if planning_grid is not None:
            self.pub_planning_grid.publish(planning_grid)

        if plan is None:
            self.get_logger().warn(f"Planning failed ({reason}). No feasible candidate path.")
            self._publish_empty_path(reason=f"planning_failed_{reason}")
            return

        best_pose = msg.poses[plan.best_pose_index]
        path_msg = self._build_path_message(plan.path_idx, final_orientation=best_pose.orientation)
        self.pub_path.publish(path_msg)
        self._current_path_idx = plan.path_idx
        self._current_include_box_lethal = include_box_lethal
        self._active_plan_mode = "candidates"
        self._active_goal_candidates = msg
        self._replan_in_progress = False
        self.get_logger().info(
            f"Published candidate path with {len(path_msg.poses)} poses (reason={reason}, candidates={len(msg.poses)}, cost={plan.cost:.2f})."
        )

    def _build_path_message(
        self,
        path_idx: List[GridIndex],
        final_yaw: Optional[float] = None,
        final_orientation=None,
    ) -> Path:
        meta = self.path_manager.meta
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self.global_frame
        if meta is None:
            return path_msg

        for (gx, gy) in path_idx:
            wx, wy = self.path_manager.grid_to_world_center(gx, gy, meta)
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = wx
            ps.pose.position.y = wy
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)

        if len(path_msg.poses) == 0:
            return path_msg

        if final_yaw is not None:
            q = quaternion_from_euler(0.0, 0.0, final_yaw)
            path_msg.poses[-1].pose.orientation.x = q[0]
            path_msg.poses[-1].pose.orientation.y = q[1]
            path_msg.poses[-1].pose.orientation.z = q[2]
            path_msg.poses[-1].pose.orientation.w = q[3]
        elif final_orientation is not None:
            path_msg.poses[-1].pose.orientation = final_orientation

        return path_msg

    def _apply_coarse_object_standoff(
        self, path_idx: List[GridIndex]
    ) -> Tuple[Optional[List[GridIndex]], Optional[float]]:
        meta = self.path_manager.meta
        if self._goal_msg is None or meta is None:
            return (None, None)

        tx = self._goal_msg.pose.position.x
        ty = self._goal_msg.pose.position.y

        radius = self.get_parameter("coarse_object_standoff").get_parameter_value().double_value
        if radius <= 0.0:
            return (None, None)

        cut_idx = self._path_index_at_radius(path_idx, tx, ty, radius, meta)
        if cut_idx is None:
            return (None, None)

        truncated = path_idx[:cut_idx + 1]
        ax, ay = self.path_manager.grid_to_world_center(truncated[-1][0], truncated[-1][1], meta)
        yaw = math.atan2(ty - ay, tx - ax)
        self.get_logger().info(
            f"Coarse object standoff applied: radius={radius:.2f}, cut_idx={cut_idx}, path_len={len(path_idx)}->{len(truncated)}"
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
            wx, wy = PathManager.grid_to_world_center(gx, gy, meta)
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
        self._current_path_idx = None
        self._replan_in_progress = False
        self.get_logger().warn(f"Published EMPTY path (reason={reason}).")

    def _get_robot_xy_in_map(self) -> Optional[Tuple[float, float]]:
        try:
            # latest available transform
            tf = self.tf_buffer.lookup_transform(self.global_frame, self.robot_frame, rclpy.time.Time())
            return (tf.transform.translation.x, tf.transform.translation.y)
        except Exception:
            return None

    def _continue_replan_check(self) -> None:
        self._replan_check_pending = False

        if self._current_path_idx is None:
            self._replan_in_progress = False
            return
        if self._active_plan_mode not in ("single", "candidates"):
            self._replan_in_progress = False
            return

        planning_grid = self.path_manager.rebuild_planning_grid(
            include_box_lethal=self._current_include_box_lethal
        )
        if planning_grid is not None:
            self.pub_planning_grid.publish(planning_grid)

        robot_xy = self._get_robot_xy_in_map()
        if self.path_manager.path_is_still_valid(self._current_path_idx, robot_xy=robot_xy):
            self.get_logger().info("Current global path is valid. --> no replanning")
            self._replan_in_progress = False
            return

        self.get_logger().warn("Current global path is blocked. Replanning.")
        if self._active_plan_mode == "single":
            self._pending_plan_mode = "single"
            self._pending_plan_reason = "path_blocked"
            self._continue_plan_and_publish(reason=self._pending_plan_reason)
        elif self._active_plan_mode == "candidates" and self._active_goal_candidates is not None:
            self._pending_goal_candidates = self._active_goal_candidates
            self._pending_plan_mode = "candidates"
            self._pending_plan_reason = "path_blocked"
            self._continue_plan_and_publish_candidates(reason=self._pending_plan_reason)
        else:
            self._replan_in_progress = False

    def _check_replan(self) -> None:
        self.get_logger().info(f'entered _check_replan function')
        if self._current_path_idx is None or self._replan_in_progress:
            return
        if self._active_plan_mode not in ("single", "candidates"):
            return

        self._replan_in_progress = True
        self._replan_check_pending = True
        self.update_object_list()


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
