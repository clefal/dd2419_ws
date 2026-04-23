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
from std_msgs.msg import String, Float32, Bool, Int64
from tf_transformations import quaternion_from_euler, euler_from_quaternion
from tf2_ros import Buffer, TransformListener
from robp_interfaces.srv import GoalsAvailable, GetClosestCube, SetStatus, GetClosestBox, OutputMapFile


from .exploration import RandomWaypointExplorer


class AutoState(Enum):
    IDLE = 'IDLE'
    INITIALIZATION = 'INITIALIZATION'
    SEARCH = 'SEARCH'
    APPROACH_OBJECT_COARSE = 'APPROACH_OBJECT_COARSE'
    APPROACH_OBJECT_FINAL = 'APPROACH_OBJECT_FINAL'
    WAIT_PICKUP_RESULT = 'WAIT_PICKUP_RESULT'
    BACKUP_BEFORE_PICKUP_RETRY = 'BACKUP_BEFORE_PICKUP_RETRY'
    RETURN_BOX_COARSE = 'RETURN_BOX_COARSE'
    RETURN_BOX_FINAL = 'RETURN_BOX_FINAL'
    BACKUP_AFTER_DROP = 'BACKUP_AFTER_DROP'
    WAIT_DROP_RESULT = 'WAIT_DROP_RESULT'


class GoalManager(Node):

    def __init__(self):
        super().__init__('goal_manager')

        self.manual_goal = False
        
        self._state = AutoState.IDLE

        self._static_loaded = False

    
        self._target_ = None   # (x, y) in fixed frame
        self._target_id = None
        self._box_id = None
        self._box_pose = None  # (x, y, yaw) in fixed frame
        self._start_x = 0.0
        self._start_y = 0.0
        self._start_yaw = 0.0


        self._goals_available = False
        self._pending_target_reason = None
        self._pending_box_reason = None
        self._pending_startup_check = False
        self._search_retarget_pending = False



        self._search_x = 1.0
        self._search_y = 2.0
        self._search_yaw = 3.1415/2
        self._active_search_goal = None
        self._explorer = RandomWaypointExplorer(
            min_step_m=1.5,
            max_step_m=2.5,
            min_revisit_dist_m=0.8,
            failed_blacklist_radius_m=0.6,
            occ_lethal=90,
            logger=self.get_logger(),
        )
        self.declare_parameter('box_standoff_distance', 0.6)

        self._fixed_frame = 'map'
        self._base_frame = 'base_link'

        self._goal_pub = self.create_publisher(PoseStamped, '/nav/goal', 10)
        self._goal_candidates_pub = self.create_publisher(PoseArray, '/nav/goal_candidates', 10)
        self._arm_status_pub = self.create_publisher(String, '/arm/action', 10)
        self._backup_pub = self.create_publisher(Float32, '/nav/backup_distance', 10)
        self._final_approach_enable_pub = self.create_publisher(Bool, '/nav/final_approach/enable', 10)
        self._final_approach_target_id_pub = self.create_publisher(Int64, '/nav/final_approach/target_id', 10)

        
        self.create_subscription(String, '/nav/status', self.status_callback, 10)
        self.create_subscription(String, '/arm/result', self.arm_result_callback, 10)
        self.create_subscription(PolygonStamped, '/workspace', self.workspace_callback, 10)
        self.create_subscription(OccupancyGrid, '/nav/planning_grid', self.planning_grid_callback, 10)
        self.create_subscription(OccupancyGrid, 'map/exploration_grid', self.exploration_grid_callback, 10)

        self.cli_goals_available = self.create_client(GoalsAvailable, 'object_manager/goals_available')
        while not self.cli_goals_available.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('goals_available service not available, waiting again...')

        self.cli_get_closest_cube = self.create_client(GetClosestCube, 'object_manager/get_closest_cube')
        while not self.cli_get_closest_cube.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('get_closest_cube service not available, waiting again...')

        self.cli_set_status = self.create_client(SetStatus, 'object_manager/set_status')
        while not self.cli_set_status.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('set_status service not available, waiting again...')

        self.cli_get_closest_box = self.create_client(GetClosestBox, 'object_manager/get_closest_box')
        while not self.cli_get_closest_box.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('get_closest_box service not available, waiting again...')


        self.cli_create_mapfile = self.create_client(OutputMapFile, 'object_manager/output_map_file')
        while not self.cli_create_mapfile.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('create_mapfile service not available, waiting again...')


        self._mapfile_timer = self.create_timer(15, self.create_mapfile)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._initial_goal_dispatched = False
        self._startup_timer = self.create_timer(0.5, self.try_startup)
        self._search_retarget_timer = self.create_timer(0.5, self.search_retarget_timer_callback)
    

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
                self.request_new_target(reason='search_retarget')
            elif not self.manual_goal and self._state == AutoState.APPROACH_OBJECT_COARSE:
                if msg.data == 'REACHED':
                    self.get_logger().info('Coarse object approach reached. Starting final approach.')
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
            elif not self.manual_goal and self._state == AutoState.BACKUP_BEFORE_PICKUP_RETRY:
                if msg.data == 'REACHED':
                    self.get_logger().info('Backup before pickup retry complete. Starting final approach again.')
                else:
                    self.get_logger().warn(
                        f'Backup before pickup retry failed with status={msg.data}. Trying final approach anyway.'
                    )
                self._retry_final_object_approach()

    # ----------------------------

    def arm_result_callback(self, msg: String):
        if self.manual_goal:
            return

        if self._state == AutoState.WAIT_PICKUP_RESULT:
            if msg.data == 'PICK_UP_SUCCESS':
                self.get_logger().info('Arm pickup succeeded. Returning to box.')

                # Remove picked cube from list (best-effort) and clear current target
                if self._target_ is not None:
                    self.set_status(reason='cube_picked') # set status of the current target to unavailable snce the pick up succeeded

                self._target_ = None
     

                self.request_box_goal_candidates(reason='pickup_success')
            elif msg.data == 'PICK_UP_FAIL_OUT_OF_REACH':
                self.get_logger().warn('Arm reported cube out of reach. Backing up before retrying final approach.')
                self._state = AutoState.BACKUP_BEFORE_PICKUP_RETRY
                self.publish_backup_distance(0.6)
            elif msg.data in (
                'PICK_UP_FAIL_NO_OBJECT',
                'PICK_UP_FAIL_NO_START',
                'PICK_UP_FAIL_NO_DETECTION',
                'PICK_UP_FAIL_TIMEOUT',
            ):
                self.get_logger().warn(f'Arm pickup failed with no detected cube: {msg.data}. Skipping target.')
                self._skip_current_target()
            elif msg.data in ('NO_HOLDING', 'PICK_UP_FAIL_NO_HOLDING', 'PICK_UP_FAIL_NO_HOLD'):
                self.get_logger().warn(f'Arm saw cube but did not grab it: {msg.data}. Retrying pickup.')
                self.publish_arm_status('PICK_UP')
            else:
                self.get_logger().info(f'Arm result received while waiting for pickup: {msg.data}')
            return

        if self._state == AutoState.WAIT_DROP_RESULT:
            if msg.data == 'DROP_SUCCESS':
                self.get_logger().info('Drop succeeded.')

                # set status of the box to available again
                if self._box_id is not None:
                    self.set_status(reason='dropoff_at_box')
       
                self._state = AutoState.BACKUP_AFTER_DROP
                self.publish_backup_distance(0.15)
                
            elif msg.data == 'DROP_FAIL_NO_OBJECT':
                self.get_logger().warn('Drop failed: DROP_FAIL_NO_OBJECT')
            else:
                self.get_logger().info(f'Arm result received while waiting for drop: {msg.data}')

    def set_status(self, reason):
        req = SetStatus.Request() 
        if reason in ('cube_picked', 'cube_skipped'):
            req.obj_id = self._target_id
            req.status = 'unavailable'  # do not select this cube as a goal again
        elif reason == 'dropoff_at_box':
            req.obj_id = self._box_id
            req.status = 'available'  # after a dropoff at box A, we have to set the status of box A back from 'isgoal' to 'available' in order to mark it as occupoied cells later in the global planner
        else:
            self.get_logger().warn(f'Unknown set_status reason: {reason}')
            return None

        return self.cli_set_status.call_async(req)

    def _retry_final_object_approach(self):
        if self._target_id is None:
            self.get_logger().warn('Cannot retry final approach because no target id is available. Returning to search.')
            self._state = AutoState.SEARCH
            self.publish_next_search_goal()
            return

        if not self.start_final_approach(self._target_id, AutoState.APPROACH_OBJECT_FINAL):
            self._state = AutoState.SEARCH
            self.publish_next_search_goal()

    def _skip_current_target(self):
        future = None
        if self._target_id is not None:
            future = self.set_status(reason='cube_skipped')

        self._target_ = None
        self._target_id = None

        if future is None:
            self._state = AutoState.SEARCH
            self.request_new_target(reason='pickup_failed_skip')
            return

        self._state = AutoState.IDLE
        future.add_done_callback(self._skip_current_target_status_callback)

    def _skip_current_target_status_callback(self, future):
        try:
            future.result()
        except Exception as e:
            self.get_logger().warn(f'Failed to mark skipped cube unavailable: {e}')
            return

        self._state = AutoState.SEARCH
        self.request_new_target(reason='pickup_failed_skip')


    def create_mapfile(self):
        req = OutputMapFile.Request()
        self.get_logger().info(f'creating mapfile entered in goal_manager')
        future = self.cli_create_mapfile.call_async(req)
        # this doesnt return anything, the output is created by the server in the object_manager.py


    def request_new_target(self, reason: str = 'unspecified'):
        if reason == 'search_retarget':
            if self._search_retarget_pending:
                return
            self._search_retarget_pending = True

        robot_xy = self.get_robot_xy()
        if robot_xy is None:
            self.get_logger().warn(f'Cannot request new target ({reason}): robot pose unavailable.')
            if reason == 'search_retarget':
                self._search_retarget_pending = False
                if self._state == AutoState.SEARCH and not self._waiting_for_result:
                    self.publish_next_search_goal()
            elif reason in ('startup', 'after_drop', 'pickup_failed_skip'):
                self._state = AutoState.SEARCH
                self.publish_next_search_goal()
            return

        self._pending_target_reason = reason
        req = GoalsAvailable.Request()
        future = self.cli_goals_available.call_async(req)
        future.add_done_callback(self.goals_available_response_callback)


    def goals_available_response_callback(self, future):
        reason = self._pending_target_reason
        try:
            res = future.result()
        except Exception as e:
            self.get_logger().error(f'goals_available call failed: {e}')
            if reason == 'search_retarget':
                self._search_retarget_pending = False
                if self._state == AutoState.SEARCH and not self._waiting_for_result:
                    self.publish_next_search_goal()
            elif reason in ('startup', 'after_drop', 'pickup_failed_skip'):
                self._state = AutoState.SEARCH
                self.publish_next_search_goal()
            return

        self._goals_available = res.goals_available
        self.get_logger().info(f'Goals available service returned: {res.goals_available}')

        if not res.goals_available:
            if reason == 'search_retarget':
                self._search_retarget_pending = False
                if self._state == AutoState.SEARCH and not self._waiting_for_result:
                    self.publish_next_search_goal()
            elif reason in ('startup', 'after_drop', 'pickup_failed_skip'):
                self._state = AutoState.SEARCH
                self.publish_next_search_goal()
            return

        robot_xy = self.get_robot_xy()
        if robot_xy is None:
            self.get_logger().warn('Robot pose unavailable after goals_available response.')
            if reason == 'search_retarget':
                self._search_retarget_pending = False
            return

        req = GetClosestCube.Request()
        req.robot_x = robot_xy[0]
        req.robot_y = robot_xy[1]
        future = self.cli_get_closest_cube.call_async(req)
        future.add_done_callback(self.get_closest_cube_response_callback)

    def get_closest_cube_response_callback(self, future):
        reason = self._pending_target_reason
        try:
            res = future.result()
        except Exception as e:
            self.get_logger().error(f'get_closest_cube Service call failed: {e}')
            if reason == 'search_retarget':
                self._search_retarget_pending = False
                if self._state == AutoState.SEARCH and not self._waiting_for_result:
                    self.publish_next_search_goal()
            elif reason in ('startup', 'after_drop', 'pickup_failed_skip'):
                self._state = AutoState.SEARCH
                self.publish_next_search_goal()
            return

        self._target_ = (res.obj_x, res.obj_y)
        self.get_logger().info(f'Closest cube to robot at {self.get_robot_xy()} is Obj{res.obj_id} at {res.obj_x}, {res.obj_y}')

        self._search_retarget_pending = False

        if (
            reason == 'search_retarget'
            and self._state == AutoState.APPROACH_OBJECT_COARSE
            and self._target_id == res.obj_id
        ):
            return

        self._target_id = res.obj_id
        self._active_search_goal = None
        self._state = AutoState.APPROACH_OBJECT_COARSE
        self.publish_goal(res.obj_x, res.obj_y, 0.0)





    def try_startup(self):
        if self.manual_goal:
            return

        if self._initial_goal_dispatched:
            return

        if self._state != AutoState.INITIALIZATION:
            return

        robot_pose = self.lookup_xy_yaw(self._fixed_frame, self._base_frame)
        if robot_pose is None:
            self.get_logger().info('Startup waiting: robot pose not available yet.')
            return

        self._start_x, self._start_y, self._start_yaw = robot_pose

        if self._pending_startup_check:
            return

        self._pending_startup_check = True
        req = GoalsAvailable.Request()
        future = self.cli_goals_available.call_async(req)
        future.add_done_callback(self.startup_goals_available_callback)

    def startup_goals_available_callback(self, future):
        self._pending_startup_check = False

        try:
            res = future.result()
        except Exception as e:
            self.get_logger().warn(f'Startup goals_available check failed: {e}')
            return

        self._goals_available = res.goals_available
        self._static_loaded = True

        if res.goals_available:
            self.get_logger().info('Startup: object_manager is ready and goals are available.')
            self.dispatch_initial_goal_after_loading()
            return

        self.get_logger().info('Startup: object_manager ready, but no goals yet. Starting search.')
        self._initial_goal_dispatched = True
        self._state = AutoState.SEARCH
        self.publish_next_search_goal()

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
    



    def dispatch_initial_goal_after_loading(self):
        if self._initial_goal_dispatched:
            return

        self._initial_goal_dispatched = True
        self.request_new_target(reason='startup')


   
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

    def exploration_grid_callback(self, msg: OccupancyGrid):
        if msg.header.frame_id and msg.header.frame_id != self._fixed_frame:
            self.get_logger().warn(
                f'Exploration grid in frame "{msg.header.frame_id}", expected "{self._fixed_frame}". Ignoring.'
            )
            return
        self._explorer.set_exploration_grid(msg)

    def search_retarget_timer_callback(self):
        if self.manual_goal:
            return
        if self._state not in (AutoState.SEARCH, AutoState.APPROACH_OBJECT_COARSE):
            self._search_retarget_pending = False
            return
        if self._search_retarget_pending:
            return

        self.request_new_target(reason='search_retarget')

    def get_robot_xy(self):
        try:
            t = self._tf_buffer.lookup_transform(
                self._fixed_frame, self._base_frame, rclpy.time.Time())
        except Exception:
            return None
        return t.transform.translation.x, t.transform.translation.y


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

    def request_box_goal_candidates(self, reason: str = 'return_box'):
        robot_xy = self.get_robot_xy()
        if robot_xy is None:
            self.get_logger().warn(f'Cannot request closest box ({reason}): robot pose unavailable.')
            self._state = AutoState.SEARCH
            self.publish_next_search_goal()
            return

        self._pending_box_reason = reason
        req = GetClosestBox.Request()
        req.robot_x = robot_xy[0]
        req.robot_y = robot_xy[1]
        future = self.cli_get_closest_box.call_async(req)
        future.add_done_callback(self.get_closest_box_response_callback)

    def get_closest_box_response_callback(self, future):
        try:
            res = future.result()
        except Exception as e:
            self.get_logger().error(f'get_closest_box service call failed: {e}')
            self._state = AutoState.SEARCH
            self.publish_next_search_goal()
            return

        self._box_id = res.obj_id
        self._box_pose = (res.obj_x, res.obj_y, res.obj_yaw)
        #self.get_logger().info(f'Closest box position received at: {self._box_pose}')

        if self._box_pose is None or self._box_id is None:
            self.get_logger().warn('No live box pose available after service response.')
            self._state = AutoState.SEARCH
            self.publish_next_search_goal()
            return

        bx, by, _ = self._box_pose
        d = float(self.get_parameter('box_standoff_distance').value)

        cands = [
            (bx + d, by),
            (bx - d, by),
            (bx, by + d),
            (bx, by - d),
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

        self._state = AutoState.RETURN_BOX_COARSE
        self._goal_candidates_pub.publish(pa)
        self._waiting_for_result = True
        self.get_logger().info(
            f'Box goal candidates sent: '
            f'c0=({cands[0][0]:.2f},{cands[0][1]:.2f}), '
            f'c1=({cands[1][0]:.2f},{cands[1][1]:.2f}), '
            f'c2=({cands[2][0]:.2f},{cands[2][1]:.2f}), '
            f'c3=({cands[3][0]:.2f},{cands[3][1]:.2f})'
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
        #self.get_logger().info(f'Final approach enable sent: {enabled}')

    def publish_final_approach_target_id(self, object_id):
        msg = Int64()
        msg.data = int(object_id)
        self._final_approach_target_id_pub.publish(msg)
        self.get_logger().info(f'Final approach target id sent: {object_id}')

 
    def _continue_after_drop(self):
        # If we still have cubes, go for the closest one; else go to SEARCH point

        self.request_new_target(reason='after_drop')

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
