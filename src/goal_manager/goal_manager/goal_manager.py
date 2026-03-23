#!/usr/bin/env python3

import json
import math
import sys
import threading
import time
from enum import Enum

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, PoseArray, PolygonStamped
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String, Float32, Bool
from tf_transformations import quaternion_from_euler, euler_from_quaternion
from tf2_ros import Buffer, TransformListener
from robp_interfaces.srv import GoalsAvailable, GetClosestCube, SetStatus, GetClosestBox


from .exploration import RandomWaypointExplorer


class AutoState(Enum):
    IDLE = 'IDLE'
    INITIALIZATION = 'INITIALIZATION'
    SEARCH = 'SEARCH'
    APPROACH_OBJECT_COARSE = 'APPROACH_OBJECT_COARSE'
    APPROACH_OBJECT_FINAL = 'APPROACH_OBJECT_FINAL'
    WAIT_PICKUP_RESULT = 'WAIT_PICKUP_RESULT'
    RETURN_BOX_COARSE = 'RETURN_BOX_COARSE'
    RETURN_BOX_FINAL = 'RETURN_BOX_FINAL'
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
        self._live_boxes = {}  # box_id -> (x, y, yaw) in fixed frame
        self._box_id = None
        self._box_pose = None  # (x, y, yaw) in fixed frame


        self._goals_available = False



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
        self._box_side_offset = 1.0  # coarse standoff along the two long sides

        self._fixed_frame = 'map'
        self._base_frame = 'base_link'

        self._goal_pub = self.create_publisher(PoseStamped, '/nav/goal', 10)
        self._goal_candidates_pub = self.create_publisher(PoseArray, '/nav/goal_candidates', 10)
        self._arm_status_pub = self.create_publisher(String, '/arm/action', 10)
        self._cubes_pub = self.create_publisher(PoseArray, '/nav/objects/cubes', 10)
        self._target_pub = self.create_publisher(PoseStamped, '/nav/target/cube', 10)
        self._box_pub = self.create_publisher(PoseStamped, '/nav/box', 10)
        self._backup_pub = self.create_publisher(Float32, '/nav/backup_distance', 10)
        self._final_approach_enable_pub = self.create_publisher(Bool, '/nav/final_approach/enable', 10)
        self._final_approach_target_id_pub = self.create_publisher(String, '/nav/final_approach/target_id', 10)
        self._object_consumed_pub = self.create_publisher(String, '/nav/object_consumed', 10)

        
        self.create_subscription(String, '/nav/status', self.status_callback, 10)
        self.create_subscription(String, '/arm/result', self.arm_result_callback, 10)
        self.create_subscription(String, 'detection/detection_manager/live_list', self.live_list_callback, 10)
        self.create_subscription(PolygonStamped, '/workspace', self.workspace_callback, 10)
        self.create_subscription(OccupancyGrid, '/nav/planning_grid', self.planning_grid_callback, 10)

        self.cli_goals_available = self.create_client(GoalsAvailable, 'object_manager/goals_available')
        while not self.cli_goals_available.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('goals_available service not available, waiting again...')

        self.cli_get_closest_cube = self.create_client(GetClosestCube, 'object_manager/get_closest_cube')
        while not self.cli_get_closest_cube.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('get_closest_cube service not available, waiting again...')

        self.cli_set_status = self.create_client(SetStatus, 'object_manager/set_status')
        while not self.cli_get_closest_cube.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('get_closest_cube service not available, waiting again...')

        self.cli_get_closest_box = self.create_client(GetClosestBox, 'object_manager/get_closest_box')
        while not self.cli_get_closest_box.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('get_closest_box service not available, waiting again...')


        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._initial_goal_dispatched = False
        self._live_list_ready = False

        self._waiting_for_result = False
        self._first_goal_delay_done = False
        self.create_timer(0.2, self.check_pending_consumed_object)

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
                    self._refresh_target_from_live_objects()
                    if not self.start_final_approach(self._target_id, AutoState.APPROACH_OBJECT_FINAL):
                        self._state = AutoState.SEARCH
                        self.publish_next_search_goal()
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
            elif not self.manual_goal and self._state == AutoState.RETURN_BOX_COARSE:
                if msg.data == 'REACHED':
                    self.get_logger().info('Coarse box approach reached. Starting final approach.')
                    if not self.start_final_approach(self._box_id, AutoState.RETURN_BOX_FINAL):
                        self.get_logger().warn('Could not start final box approach. Returning to search.')
                        self._state = AutoState.SEARCH
                        self.publish_next_search_goal()
                else:
                    self.get_logger().warn('Return-to-box coarse approach failed. Drop command will not be sent.')
            elif not self.manual_goal and self._state == AutoState.RETURN_BOX_FINAL:
                if msg.data == 'REACHED':
                    self.get_logger().info('Final box approach reached. Triggering arm drop.')
                    self.stop_final_approach()
                    self.publish_arm_status('DROP')
                    self._state = AutoState.WAIT_DROP_RESULT
                else:
                    self.get_logger().warn('Final box approach failed. Drop command will not be sent.')
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
            if msg.data == 'PICK_UP_SUCCESS':
                self.get_logger().info('Arm pickup succeeded. Returning to box.')

                # Remove picked cube from list (best-effort) and clear current target
                if self._target_ is not None:
                    self.set_status() # set status of the current target to unavailable snce the pick up succeeded

                self._target_ = None
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

    def set_status(self):
        req = SetStatus.Request()
        req.obj_id = self._target_id
        req.status = 'unavailable'  # we currently use this function to set the status of an object_id to unavailable after we picked it up
        future = self.cli_set_status.call_async(req)
        # i think we dont need an done_callback here because we dont return anything...




    def publish_new_target(self):
        '''Checks if targets are available, if yes then it sets self._target_ to the closest one and, returns True if successful '''
        # service: get_closest_cube
        req_goals_available = GoalsAvailable.Request()
        # Send the request asynchronously
        future_goals_available = self.cli_goals_available.call_async(req_goals_available)
        # Attach a callback function that will run ONLY when the response arrives.
        future_goals_available.add_done_callback(self.goals_available_response_callback)

        if not self._goals_available:
            self.get_logger().warn(f'No goals available during publish_new_target(). cli_goals_available service returned False')
            return False

        prev_target = self._target_
        req_closest_goal = GetClosestCube.Request()
        req_closest_goal.robot_x, req_closest_goal.robot_y =self.get_robot_xy()
        future_closest_goal = self.cli_get_closest_cube.call_async(req_closest_goal)
        future_closest_goal.add_done_callback(self.get_closest_cube_response_callback)
        new_target = self._target_
        
        if prev_target == new_target:
            self.get_logger().warn(f'during publish_new_target: prev_target = new_target ->service didnt return new target')
            return False
        
        return True


    def goals_available_response_callback(self, future):
        """This function is triggered automatically when the goals_avaibalbe service responds."""
        try:
            # Extract the actual response from the Future object
            res = future.result()
            self.get_logger().info(f'Goals available service returned: {res.goals_available}')
            self._goals_available = res.goals_available
        except Exception as e:
            # It's good practice to catch exceptions in case the service server crashed or failed
            self.get_logger().error(f'goals_available call failed: {e}')

    def get_closest_cube_response_callback(self, future):
        try:
            res = future.result()
            self._target_ = (res.obj_x, res.obj_y)
            self._target_id = res.obj_id
            self.get_logger().info(f'closest cube to robot at: {self.get_robot_xy()} is Obj{res.obj_id} at {res.obj_x}, {res.obj_y}')
            self.publish_goal(res.obj_x, res.obj_y, 0.0)
            
        except Exception as e:
            self.get_logger().error(f'get_closest_cube Service call failed: {e}')


    def load_robot_inital_pose(self):
        if self._static_loaded:
            return
        #start pose (robot in map)
        robot_pose = self.lookup_xy_yaw(self._fixed_frame, self._base_frame)
        if robot_pose is not None:
            self._start_x, self._start_y, self._start_yaw = robot_pose
            self.get_logger().info(f'Loaded start pose (robot in map): x={self._start_x:.2f}, y={self._start_y:.2f}, yaw={self._start_yaw:.2f}')

        self._static_loaded = True

        # Startup gate: do not dispatch first nav goal until static workspace/map frames are loaded.
        if (not self.manual_goal) and (not self._initial_goal_dispatched):
            self.dispatch_initial_goal_after_loading()


    def dispatch_initial_goal_after_loading(self):
        if self._initial_goal_dispatched:
            return

        success = self.publish_new_target()

        if success: 
            self._state = AutoState.APPROACH_OBJECT_COARSE
            tx, ty = self._target_
            self.get_logger().info(
                f'Initial target object selected at x={tx:.2f}, y={ty:.2f}. Publishing coarse goal.'
            )

        # check target to check if we succeded with the publish_new_target
        if self._target_ is not None:
            self._initial_goal_dispatched = True
            return

        # No cubes known: begin exploration search.
        self._state = AutoState.SEARCH
        self.publish_next_search_goal()
        self.get_logger().info('State SEARCH: starting random exploratory search after first live_list update.')
        self._initial_goal_dispatched = True


   
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

        if self._box_pose is not None:  #TODO Is this still needed?
            bx, by, byaw = self._box_pose
            box_msg = PoseStamped()
            box_msg.header.stamp = pa.header.stamp
            box_msg.header.frame_id = self._fixed_frame
            box_msg.pose.position.x = float(bx)
            box_msg.pose.position.y = float(by)
            box_msg.pose.position.z = 0.0
            q = quaternion_from_euler(0.0, 0.0, byaw)
            box_msg.pose.orientation.x = q[0]
            box_msg.pose.orientation.y = q[1]
            box_msg.pose.orientation.z = q[2]
            box_msg.pose.orientation.w = q[3]
            self._box_pub.publish(box_msg)
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
            if object_id.startswith('object_')
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

    def select_box_id(self):
        box_candidates = sorted(self._live_boxes.keys()) if hasattr(self, '_live_boxes') else []
        if len(box_candidates) == 0:
            return None
        return box_candidates[0]

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

    def _extract_box_yaw(self, item):
        # TODO: align this with the exact detection_manager live_list box orientation fields once merged.
        if 'yaw' in item:
            return float(item['yaw'])
        if 'theta' in item:
            return float(item['theta'])
        orientation = item.get('orientation')
        if isinstance(orientation, dict):
            if 'yaw' in orientation:
                return float(orientation['yaw'])
            if all(k in orientation for k in ('x', 'y', 'z', 'w')):
                q = orientation
                return euler_from_quaternion([q['x'], q['y'], q['z'], q['w']])[2]
        return 0.0

    def _update_box_from_live_list(self, live_boxes):
        self._live_boxes = dict(live_boxes)
        self._box_id = self.select_box_id()
        if self._box_id is None:
            self._box_pose = None
            return
        self._box_pose = self._live_boxes.get(self._box_id)

    def start_final_approach(self, target_id, next_state):
        if target_id is None:
            self.get_logger().warn('Cannot start final approach because no live target is available.')
            return False

        self._state = next_state
        self.publish_final_approach_target_id(target_id)
        self.publish_final_approach_enable(True)
        self._waiting_for_result = True
        return True

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

        req = GetClosestBox.Request()
        req.robot_x, req.robot_y = self.get_robot_xy()
        future = self.cli_get_closest_box.call_async(req)
        future.add_done_callback(self.get_closest_box_response_callback)

        if self._box_pose is None or self._box_id is None:
            self.get_logger().warn('No live box pose available. Cannot publish box goal candidates.')
            self._state = AutoState.SEARCH
            self.publish_next_search_goal()
            return

        bx, by, byaw = self._box_pose

        # Approach box from its two long sides (left/right in box local frame).
        axis_yaw = byaw + math.pi / 2.0
        ux = math.cos(axis_yaw)
        uy = math.sin(axis_yaw)

        cands = [
            (bx + self._box_side_offset * ux, by + self._box_side_offset * uy),
            (bx - self._box_side_offset * ux, by - self._box_side_offset * uy),
        ]

        pa = PoseArray()
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.header.frame_id = self._fixed_frame

        for (gx, gy) in cands:
            pose = PoseStamped().pose
            pose.position.x = float(gx)
            pose.position.y = float(gy)
            pose.position.z = 0.0
            yaw_to_box = math.atan2(by - gy, bx - gx)
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

    def get_closest_box_response_callback(self, future):
        res = future.result()
        self._box_id = res.obj_id
        self._box_pose = (res.obj_x, res.obj_y, res.obj_yaw)
        self.get_logger().info(f'Closest box position recieved at: {self._box_pose}')

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

    def finish_consumed_object_handshake(self):
        self.get_logger().info('Consumed object removed from live_list. Proceeding to box return.')
        self._pending_consumed_object_id = None
        self._pending_consumed_deadline = None
        self._target_ = None
        self._target_id = None
        self.publish_topics()

        self._state = AutoState.RETURN_BOX_COARSE
        self.publish_box_goal_candidates()

    def check_pending_consumed_object(self):
        if self._pending_consumed_object_id is None:
            return

        if self._pending_consumed_deadline is None:
            return

        if time.time() <= self._pending_consumed_deadline:
            return

        self.get_logger().warn(
            f'Consumed object {self._pending_consumed_object_id} still present after timeout. Continuing anyway.'
        )
        self.finish_consumed_object_handshake()

    def _continue_after_drop(self):
        # If we still have cubes, go for the closest one; else go to SEARCH point

        success = self.publish_new_target()
        if success:
            tx, ty = self._target_
            self._state = AutoState.APPROACH_OBJECT_COARSE
            self.get_logger().info(
                f'Target object selected at x={tx:.2f}, y={ty:.2f}. Publishing coarse goal.'
            )
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
