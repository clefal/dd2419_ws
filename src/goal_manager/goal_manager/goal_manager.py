#!/usr/bin/env python3

import json
import math
import sys
import threading
import time
from enum import Enum

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, PointStamped, PoseArray, PolygonStamped
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String, Float32, Bool
from tf_transformations import quaternion_from_euler, euler_from_quaternion
from tf2_ros import Buffer, TransformListener

from .exploration import RandomWaypointExplorer


class AutoState(Enum):
    IDLE = 'IDLE'
    INITIALIZATION = 'INITIALIZATION'
    SEARCH = 'SEARCH'
    APPROACH_OBJECT_COARSE = 'APPROACH_OBJECT_COARSE'
    APPROACH_OBJECT_FINAL = 'APPROACH_OBJECT_FINAL'
    WAIT_PICKUP_RESULT = 'WAIT_PICKUP_RESULT'
    RETURN_HOME = 'RETURN_HOME'
    BACKUP_AFTER_DROP = 'BACKUP_AFTER_DROP'
    WAIT_DROP_RESULT = 'WAIT_DROP_RESULT'


class GoalManager(Node):

    def __init__(self):
        super().__init__('goal_manager')

        self.manual_goal = False
        
        self._state = AutoState.IDLE
        self._latest_cube = None
        self._detection_locked = False
        self._approach_distance = 0.17

        self._merge_radius = 0.10  # m, deduplicate detections
        self._cubes = []      # list of (x, y) in fixed frame
        self._target_ = None   # (x, y) in fixed frame
        self._target_id = None
        self._live_objects = {}  # object_id -> (x, y) in fixed frame
        self._consumed_object_ids = set()  # locally ignore consumed objects until detection_manager reflects removal


        self._search_x = 1.0
        self._search_y = 2.0
        self._search_yaw = 3.1415/2
        self._active_search_goal = None
        self._explorer = RandomWaypointExplorer(
            min_step_m=1.0,
            max_step_m=2.0,
            min_revisit_dist_m=0.8,
            failed_blacklist_radius_m=0.6,
            occ_lethal=90,
        )
        self._home_x = 0.0
        self._home_y = 0.0
        self._home_yaw = 0.0
        self._box_side_offset = 0.18    #TODO

        self._fixed_frame = 'map'
        self._base_frame = 'base_link'

        self._object_frame_prefix = 'object'
        self._box_frame = 'box'
        self._max_static_objects = 50

        self._start_x = 0.0
        self._start_y = 0.0
        self._start_yaw = 0.0


        self._goal_pub = self.create_publisher(PoseStamped, '/nav/goal', 10)
        self._goal_candidates_pub = self.create_publisher(PoseArray, '/nav/goal_candidates', 10)
        self._arm_status_pub = self.create_publisher(String, '/arm/action', 10)
        self._cubes_pub = self.create_publisher(PoseArray, '/nav/objects/cubes', 10)
        self._target_pub = self.create_publisher(PoseStamped, '/nav/target/cube', 10)
        self._backup_pub = self.create_publisher(Float32, '/nav/backup_distance', 10)
        self._final_approach_enable_pub = self.create_publisher(Bool, '/nav/final_approach/enable', 10)
        self._final_approach_target_id_pub = self.create_publisher(String, '/nav/final_approach/target_id', 10)
        self._object_consumed_pub = self.create_publisher(String, '/nav/object_consumed', 10)

        
        self.create_subscription(String, '/nav/status', self.status_callback, 10)
        self.create_subscription(String, '/arm/result', self.arm_result_callback, 10)
        #self.create_subscription(PointStamped, '/detection/objects/green_cube', self.cube_callback, 10)
        #self.create_subscription(PointStamped, '/detection/objects/red_cube', self.cube_callback, 10)
        #self.create_subscription(PointStamped, '/detection/objects/blue_cube', self.cube_callback, 10)
        self.create_subscription(String, 'detection/detection_manager/live_list', self.live_list_callback, 10)
        self.create_subscription(PolygonStamped, '/workspace', self.workspace_callback, 10)
        self.create_subscription(OccupancyGrid, '/nav/planning_grid', self.planning_grid_callback, 10)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._static_loaded = False
        self._initial_goal_dispatched = False
        self.create_timer(0.5, self.try_load_static_frames_once)

        self._waiting_for_result = False
        self._first_goal_delay_done = False

        if self.manual_goal:
            thread = threading.Thread(target=self.manual_input_loop, daemon=True)
            thread.start()
        else:
            self.start_autonomous_sequence()

    # ----------------------------

    def status_callback(self, msg: String):
        if not self._waiting_for_result:
            return

        if msg.data in ('REACHED', 'FAILED'):
            self._waiting_for_result = False
            if not self.manual_goal and self._state == AutoState.SEARCH:
                self.get_logger().info(f'Search goal finished with status={msg.data}')
                if self._active_search_goal is not None:
                    self._explorer.note_waypoint_result(self._active_search_goal, msg.data)
                    self._active_search_goal = None
                self.publish_next_search_goal()
            elif not self.manual_goal and self._state == AutoState.APPROACH_OBJECT_COARSE:
                if msg.data == 'REACHED':
                    self.get_logger().info('Coarse object approach reached. Starting final approach.')
                    self.start_final_approach()
                else:
                    self.get_logger().warn('Coarse object approach failed. Returning to search.')
                    self.stop_final_approach()
                    self._state = AutoState.SEARCH
                    self.publish_next_search_goal()
            elif not self.manual_goal and self._state == AutoState.APPROACH_OBJECT_FINAL:
                if msg.data == 'REACHED':
                    self.get_logger().info('Final object approach reached. Triggering arm pickup.')
                    self.stop_final_approach()
                    self.publish_arm_status('PICK_UP')
                    self._state = AutoState.WAIT_PICKUP_RESULT
                else:
                    self.get_logger().warn('Final object approach failed. Returning to search.')
                    self.stop_final_approach()
                    self._state = AutoState.SEARCH
                    self.publish_next_search_goal()
            elif not self.manual_goal and self._state == AutoState.RETURN_HOME:
                if msg.data == 'REACHED':
                    self.get_logger().info('Home reached. Triggering arm drop.')
                    self.publish_arm_status('DROP')
                    self._state = AutoState.WAIT_DROP_RESULT
                else:
                    self.get_logger().warn('Return-home goal failed. Drop command will not be sent.')
            elif not self.manual_goal and self._state == AutoState.BACKUP_AFTER_DROP:
                if msg.data == 'REACHED':
                    self.get_logger().info('Backup after drop complete.')
                    self._continue_after_drop()
                else:
                    self.get_logger().warn(f'Backup after drop failed with status={msg.data}. Continuing anyway.')
                    self._continue_after_drop()

    # ----------------------------

    def arm_result_callback(self, msg: String):
        if self.manual_goal:
            return

        if self._state == AutoState.WAIT_PICKUP_RESULT:
            if msg.data == 'PICK_UP_SUCCESS':   #TODO this should be verified with arm camera later -> publish consumed objects
                self.get_logger().info('Arm pickup succeeded. Returning home.')
                # detection_manager owns the authoritative live object list.
                # We only notify it and locally ignore the consumed target until its next update reflects removal.
                if self._target_id is not None:
                    self.publish_object_consumed(self._target_id)
                    self._consumed_object_ids.add(self._target_id)
                if self._target_ is not None:
                    tx, ty = self._target_
                    self._cubes = [(x, y) for (x, y) in self._cubes if math.hypot(x - tx, y - ty) > self._merge_radius]
                self._target_ = None
                self._target_id = None
                self.publish_topics()

                self._state = AutoState.RETURN_HOME
                self.publish_box_goal_candidates()
            elif msg.data in ('PICK_UP_FAIL_NO_OBJECT', 'PICK_UP_FAIL_NO_START'):
                self.get_logger().warn(f'Arm pickup failed: {msg.data}')
            else:
                self.get_logger().info(f'Arm result received while waiting for pickup: {msg.data}')
            return

        if self._state == AutoState.WAIT_DROP_RESULT:
            if msg.data == 'DROP_SUCCESS':
                self.get_logger().info('Drop succeeded.')

                self._detection_locked = False
                self._state = AutoState.BACKUP_AFTER_DROP
                self.publish_backup_distance(0.15)
                
            elif msg.data == 'DROP_FAIL_NO_OBJECT':
                self.get_logger().warn('Drop failed: DROP_FAIL_NO_OBJECT')
            else:
                self.get_logger().info(f'Arm result received while waiting for drop: {msg.data}')

    def live_list_callback(self, msg: String):
        # TODO: replace std_msgs/String with the detection_manager live_list message once it lands.
        live_objects = self._parse_live_list(msg.data)
        if live_objects is None:
            return

        self._live_objects = dict(live_objects)
        self._consumed_object_ids.intersection_update(self._live_objects.keys())
        self._sync_cubes_from_live_objects()
        self._refresh_target_from_live_objects()
        self.publish_topics()

        if self.manual_goal:
            return

        if self._state in (AutoState.RETURN_HOME, AutoState.WAIT_PICKUP_RESULT, AutoState.APPROACH_OBJECT_FINAL):
            return

        if not self._static_loaded or self._waiting_for_result:
            return

        if self._state not in (AutoState.SEARCH, AutoState.INITIALIZATION):
            return

        best_id = self.select_closest_object_id()
        if best_id is None:
            return

        if self._target_id == best_id and self._target_ is not None:
            return

        self._target_id = best_id
        self._refresh_target_from_live_objects()
        if self._target_ is None:
            return

        self.publish_topics()  # publish updated target immediately

        tx, ty = self._target_
        # Enter coarse approach state; final approach starts after the 1 m standoff is reached.
        self._active_search_goal = None
        self._state = AutoState.APPROACH_OBJECT_COARSE
        self.get_logger().info(
            f'Target object {self._target_id} at x={tx:.2f}, y={ty:.2f}. Publishing coarse goal.'
        )
        self.publish_goal(tx, ty, 0.0)

    def _parse_live_list(self, payload: str):
        if not payload.strip():
            return {}

        try:
            data = json.loads(payload)
        except Exception:
            self.get_logger().warn('Failed to parse detection_manager live_list payload. Keeping previous objects.')
            return None

        tracked = {}

        if isinstance(data, dict) and 'objects' in data and isinstance(data['objects'], list):
            iterable = data['objects']
        elif isinstance(data, list):
            iterable = data
        elif isinstance(data, dict):
            iterable = []
            for object_id, item in data.items():
                if isinstance(item, dict):
                    entry = dict(item)
                    entry['id'] = object_id
                    iterable.append(entry)
        else:
            iterable = []

        for item in iterable:
            if not isinstance(item, dict):
                continue
            object_id = str(item.get('id', '')).strip()
            if object_id == '':
                continue
            try:
                x = float(item['x'])
                y = float(item['y'])
            except Exception:
                continue
            tracked[object_id] = (x, y)

        return tracked

    def lookup_xy_yaw(self, parent_frame: str, child_frame: str):
        try:
            t = self._tf_buffer.lookup_transform(parent_frame, child_frame, rclpy.time.Time())
        except Exception:
            return None
        x = t.transform.translation.x
        y = t.transform.translation.y
        q = t.transform.rotation
        yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        return (x, y, yaw)
    

    def try_load_static_frames_once(self):
        if self._static_loaded:
            return

        # 1) box -> set home
        box_pose = self.lookup_xy_yaw(self._fixed_frame, self._box_frame)
        if box_pose is None:
            # workspace_loader not ready yet
            return

        bx, by, byaw = box_pose
        self._home_x, self._home_y, self._home_yaw = bx, by, byaw
        self.get_logger().info(f'Loaded box frame as home: x={bx:.2f}, y={by:.2f}, yaw={byaw:.2f}')

        # 2) start pose (robot in map)
        robot_pose = self.lookup_xy_yaw(self._fixed_frame, self._base_frame)
        if robot_pose is not None:
            self._start_x, self._start_y, self._start_yaw = robot_pose
            self.get_logger().info(f'Loaded start pose (robot in map): x={self._start_x:.2f}, y={self._start_y:.2f}, yaw={self._start_yaw:.2f}')

        # 3) objects -> seed cube list
        seeded = 0
        for i in range(self._max_static_objects):
            child = f'{self._object_frame_prefix}{i}'
            obj_pose = self.lookup_xy_yaw(self._fixed_frame, child)
            if obj_pose is None:
                # assume contiguous indices; stop at first missing
                break
            ox, oy, _ = obj_pose
            # Seed known objects from static TF (deduplicated by merge radius).
            already_known = any(
                math.hypot(ox - cx, oy - cy) <= self._merge_radius for (cx, cy) in self._cubes
            )
            if not already_known:
                self._cubes.append((ox, oy))
                self._live_objects[f'object_{i}'] = (ox, oy)
            seeded += 1

        if seeded > 0:
            self.get_logger().info(f'Seeded {seeded} cubes from static TF frames ({self._object_frame_prefix}0..).')
            self.publish_topics()
        
        self._static_loaded = True

        # Startup gate: do not dispatch first nav goal until static workspace/map frames are loaded.
        if (not self.manual_goal) and (not self._initial_goal_dispatched):
            self.dispatch_initial_goal_after_loading()
    # ----------------------------

    def dispatch_initial_goal_after_loading(self):
        if self._initial_goal_dispatched:
            return

        # If cubes are known, start with closest cube approach.
        if len(self._cubes) > 0:
            best_id = self.select_closest_object_id()
            if best_id is not None:
                self._target_id = best_id
                self._refresh_target_from_live_objects()
            robot_xy = self.get_robot_xy()
            if robot_xy is not None and self._target_ is not None:
                self.publish_topics()
                tx, ty = self._target_
                self._state = AutoState.APPROACH_OBJECT_COARSE
                self.get_logger().info(
                    f'Initial target object selected at x={tx:.2f}, y={ty:.2f}. Publishing coarse goal.'
                )
                self.publish_goal(tx, ty, 0.0)
                self._initial_goal_dispatched = True
                return
            self.get_logger().warn('Cubes known at startup, but robot pose unavailable. Falling back to search.')

        # No cubes known: begin exploration search.
        self._state = AutoState.SEARCH
        self.publish_next_search_goal()
        self.get_logger().info('State SEARCH: starting random exploratory search after workspace load.')
        self._initial_goal_dispatched = True



    def point_to_fixed_xy(self, msg: PointStamped):
        """
        Transform PointStamped into self._fixed_frame using TF.
        Returns (x, y) in fixed frame or None.
        """
        try:
            t = self._tf_buffer.lookup_transform(
                self._fixed_frame,
                msg.header.frame_id,
                rclpy.time.Time()
            )
        except Exception:
            return None

        tx = t.transform.translation.x
        ty = t.transform.translation.y
        q = t.transform.rotation
        yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]

        # rotate + translate
        x_local = msg.point.x
        y_local = msg.point.y
        x = tx + (x_local * math.cos(yaw) - y_local * math.sin(yaw))
        y = ty + (x_local * math.sin(yaw) + y_local * math.cos(yaw))
        return (x, y)

    def workspace_callback(self, msg: PolygonStamped):
        if msg.header.frame_id and msg.header.frame_id != self._fixed_frame:
            self.get_logger().warn(
                f'Workspace polygon in frame "{msg.header.frame_id}", expected "{self._fixed_frame}". Ignoring.'
            )
            return

        pts = [(float(p.x), float(p.y)) for p in msg.polygon.points]
        if len(pts) < 3:
            self.get_logger().warn('Workspace polygon has fewer than 3 points. Ignoring.')
            return

        self._explorer.set_workspace_polygon(pts)
        self.get_logger().info(f'Workspace polygon loaded ({len(pts)} points).')

        # If we are currently searching without motion, kick patrol immediately.
        if self._state == AutoState.SEARCH and not self._waiting_for_result:
            self.publish_next_search_goal()

    def planning_grid_callback(self, msg: OccupancyGrid):
        self._explorer.set_planning_grid(msg)


    def publish_topics(self):
        # Publish cube list
        pa = PoseArray()
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.header.frame_id = self._fixed_frame
        for (x, y) in self._cubes:
            p = PoseStamped()
            # PoseArray stores Pose, so create Pose then append
            pose = PoseStamped().pose
            pose.position.x = float(x)
            pose.position.y = float(y)
            pose.position.z = 0.0
            pose.orientation.w = 1.0
            pa.poses.append(pose)
        self._cubes_pub.publish(pa)

        # Publish current target cube pose (if any)
        if self._target_ is not None:
            tx, ty = self._target_
            tgt = PoseStamped()
            tgt.header.stamp = pa.header.stamp
            tgt.header.frame_id = self._fixed_frame
            tgt.pose.position.x = float(tx)
            tgt.pose.position.y = float(ty)
            tgt.pose.position.z = 0.0
            tgt.pose.orientation.w = 1.0
            self._target_pub.publish(tgt)


    def cube_callback(self, msg: PointStamped):
        # TODO: remove this fallback when detection_manager/live_list fully replaces point detections.
        if self.manual_goal:
            return

        # Accept detections in SEARCH, coarse approach, RETURN_HOME
        if self._state not in (AutoState.SEARCH, AutoState.APPROACH_OBJECT_COARSE, AutoState.RETURN_HOME, AutoState.INITIALIZATION): #added Initialization as a state in which we detect objects
            return

        obj_xy = self.point_to_fixed_xy(msg)
        if obj_xy is None:
            self.get_logger().warn('Cube detected, but TF is unavailable. Ignoring detection.')
            return

        ox, oy = obj_xy

        # Deduplicate by merge radius
        is_new = True
        for i, (cx, cy) in enumerate(self._cubes):
            if math.hypot(ox - cx, oy - cy) <= self._merge_radius:
                # Same cube: update stored position (simple replace)
                self._cubes[i] = (ox, oy)
                is_new = False
                break

        if is_new:
            self._cubes.append((ox, oy))
            object_id = f'object_{len(self._live_objects)}'
            self._live_objects[object_id] = (ox, oy)
            self.get_logger().info(f'New cube added at x={ox:.2f}, y={oy:.2f} (total={len(self._cubes)})')

        # Always publish cube topics so planner can avoid them (even during RETURN_HOME)
        self.publish_topics()

    # ----------------------------

    def get_robot_xy(self):
        try:
            t = self._tf_buffer.lookup_transform(
                self._fixed_frame, self._base_frame, rclpy.time.Time())
        except Exception:
            return None
        return t.transform.translation.x, t.transform.translation.y

    def select_closest_object_id(self):
        object_candidates = [
            (object_id, xy) for object_id, xy in self._live_objects.items()
            if object_id.startswith('object_') and object_id not in self._consumed_object_ids
        ]
        if len(object_candidates) == 0:
            return None

        robot_xy = self.get_robot_xy()
        if robot_xy is None:
            return object_candidates[0][0]

        rx, ry = robot_xy
        best_id, _ = min(
            object_candidates,
            key=lambda item: math.hypot(item[1][0] - rx, item[1][1] - ry),
        )
        return best_id

    def _refresh_target_from_live_objects(self):
        if self._target_id is None:
            self._target_ = None
            return

        self._target_ = self._live_objects.get(self._target_id)

    def _sync_cubes_from_live_objects(self):
        self._cubes = [
            xy for object_id, xy in self._live_objects.items()
            if object_id.startswith('object_')
        ]

    def start_final_approach(self):
        self._refresh_target_from_live_objects()
        if self._target_id is None or self._target_ is None:
            self.get_logger().warn('Cannot start final approach because no live target is available.')
            self._state = AutoState.SEARCH
            self.publish_next_search_goal()
            return

        self._state = AutoState.APPROACH_OBJECT_FINAL
        self.publish_final_approach_target_id(self._target_id)
        self.publish_final_approach_enable(True)
        self._waiting_for_result = True

    def stop_final_approach(self):
        self.publish_final_approach_enable(False)

    def publish_next_search_goal(self):
        if self._state != AutoState.SEARCH:
            return

        robot_xy = self.get_robot_xy()
        if robot_xy is None:
            self.get_logger().warn('Robot pose unavailable. Cannot pick exploratory waypoint.')
            return

        wx = self._explorer.next_waypoint(robot_xy)
        if wx is None:
            gx, gy, gyaw = self._search_x, self._search_y, self._search_yaw
            self._active_search_goal = (gx, gy)
            self.get_logger().warn(
                'Explorer could not sample valid waypoint. Falling back to fixed search point.'
            )
        else:
            gx, gy = wx
            rx, ry = robot_xy
            gyaw = math.atan2(gy - ry, gx - rx)
            self._active_search_goal = (gx, gy)
            self._explorer.note_waypoint_dispatched((gx, gy))

        self.publish_goal(gx, gy, gyaw)

    # ----------------------------

    def publish_goal(self, gx, gy, gyaw=0.0):
        if not self._first_goal_delay_done:
            self.get_logger().info('Waiting 3.0s before sending first goal.')
            time.sleep(3.0)
            self._first_goal_delay_done = True

        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = self._fixed_frame

        goal.pose.position.x = float(gx)
        goal.pose.position.y = float(gy)
        goal.pose.position.z = 0.0

        q = quaternion_from_euler(0.0, 0.0, gyaw)
        goal.pose.orientation.x = q[0]
        goal.pose.orientation.y = q[1]
        goal.pose.orientation.z = q[2]
        goal.pose.orientation.w = q[3]

        self._goal_pub.publish(goal)
        self._waiting_for_result = True

        self.get_logger().info(f'Goal sent: x={gx:.2f}, y={gy:.2f}, yaw={gyaw:.2f}')

    def publish_box_goal_candidates(self):
        # Approach box from its two long sides (left/right in box local frame).
        axis_yaw = self._home_yaw + math.pi / 2.0
        ux = math.cos(axis_yaw)
        uy = math.sin(axis_yaw)

        cands = [
            (self._home_x + self._box_side_offset * ux, self._home_y + self._box_side_offset * uy),
            (self._home_x - self._box_side_offset * ux, self._home_y - self._box_side_offset * uy),
        ]

        pa = PoseArray()
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.header.frame_id = self._fixed_frame

        for (gx, gy) in cands:
            pose = PoseStamped().pose
            pose.position.x = float(gx)
            pose.position.y = float(gy)
            pose.position.z = 0.0
            yaw_to_box = math.atan2(self._home_y - gy, self._home_x - gx)
            q = quaternion_from_euler(0.0, 0.0, yaw_to_box)
            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]
            pa.poses.append(pose)

        self._goal_candidates_pub.publish(pa)
        self._waiting_for_result = True
        self.get_logger().info(
            f'Box goal candidates sent: '
            f'c0=({cands[0][0]:.2f},{cands[0][1]:.2f}), '
            f'c1=({cands[1][0]:.2f},{cands[1][1]:.2f})'
        )

    # ----------------------------

    def publish_arm_status(self, text: str):
        msg = String()
        msg.data = text
        self._arm_status_pub.publish(msg)
        self.get_logger().info(f'Arm status command sent: {text}')

    def publish_backup_distance(self, meters: float):
        msg = Float32()
        msg.data = float(meters)
        self._backup_pub.publish(msg)
        self._waiting_for_result = True
        self.get_logger().info(f'Backup command sent: {meters:.2f} m')

    def publish_final_approach_enable(self, enabled: bool):
        msg = Bool()
        msg.data = bool(enabled)
        self._final_approach_enable_pub.publish(msg)
        self.get_logger().info(f'Final approach enable sent: {enabled}')

    def publish_final_approach_target_id(self, object_id: str):
        msg = String()
        msg.data = object_id
        self._final_approach_target_id_pub.publish(msg)
        self.get_logger().info(f'Final approach target id sent: {object_id}')

    def publish_object_consumed(self, object_id: str):
        msg = String()
        msg.data = object_id
        self._object_consumed_pub.publish(msg)
        self.get_logger().info(f'Object consumed feedback sent: {object_id}')

    def _continue_after_drop(self):
        # If we still have cubes, go for the closest one; else go to SEARCH point
        best_id = self.select_closest_object_id()
        if best_id is not None:
            self._target_id = best_id
            self._refresh_target_from_live_objects()
            if self._target_ is not None:
                self.publish_topics()

                tx, ty = self._target_
                self._state = AutoState.APPROACH_OBJECT_COARSE
                self.get_logger().info(
                    f'Target object selected at x={tx:.2f}, y={ty:.2f}. Publishing coarse goal.'
                )
                self.publish_goal(tx, ty, 0.0)
                return

        # No cubes known: return to SEARCH
        self._state = AutoState.SEARCH
        self.publish_next_search_goal()
        self.get_logger().info('State SEARCH: continuing exploratory patrol while detection runs.')

    # ----------------------------

    def start_autonomous_sequence(self):
        self._state = AutoState.INITIALIZATION
        self.publish_arm_status('START')
        self.get_logger().info('State INITIALIZATION: waiting for workspace/static frames before first goal.')

    # ----------------------------

    def manual_input_loop(self):
        self.get_logger().info('Manual goal mode: type "x y yaw" and press Enter')

        while rclpy.ok():
            try:
                line = sys.stdin.readline()
                if not line:
                    continue

                parts = line.strip().split()
                if len(parts) != 3:
                    print('Enter: x y yaw')
                    continue

                gx = float(parts[0])
                gy = float(parts[1])
                gyaw = math.radians(float(parts[2]))

                if self._waiting_for_result:
                    print('Robot still moving — wait for REACHED/FAILED')
                    continue

                self.publish_goal(gx, gy, gyaw)

            except Exception:
                print('Invalid input. Use: x y yaw')

# ----------------------------

def main():
    rclpy.init()
    node = GoalManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == '__main__':
    main()
