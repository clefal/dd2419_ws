#!/usr/bin/env python3

import math
import sys
import threading
from enum import Enum

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, PointStamped, PoseArray
from std_msgs.msg import String
from tf_transformations import quaternion_from_euler, euler_from_quaternion
from tf2_ros import Buffer, TransformListener


class AutoState(Enum):
    IDLE = 'IDLE'
    INITIALIZATION = 'INITIALIZATION'
    SEARCH = 'SEARCH'
    APPROACH_OBJECT = 'APPROACH_OBJECT'
    WAIT_PICKUP_RESULT = 'WAIT_PICKUP_RESULT'
    RETURN_HOME = 'RETURN_HOME'
    WAIT_DROP_RESULT = 'WAIT_DROP_RESULT'


class GoalManager(Node):

    def __init__(self):
        super().__init__('goal_manager')

        self.manual_goal = False
        
        self._state = AutoState.IDLE
        self._latest_cube = None
        self._detection_locked = False
        self._approach_distance = 0.18

        self._merge_radius = 0.10  # m, deduplicate detections
        self._cubes = []      # list of (x, y) in fixed frame
        self._target_ = None   # (x, y) in fixed frame


        self._search_x = 3.0
        self._search_y = 0.0
        self._search_yaw = 0.0
        self._home_x = 0.0
        self._home_y = 0.0
        self._home_yaw = 0.0

        self._fixed_frame = 'map'
        self._base_frame = 'base_link'

        self._object_frame_prefix = 'object'
        self._box_frame = 'box'
        self._max_static_objects = 50

        self._start_x = 0.0
        self._start_y = 0.0
        self._start_yaw = 0.0


        self._goal_pub = self.create_publisher(PoseStamped, '/nav/goal', 10)
        self._arm_status_pub = self.create_publisher(String, '/arm/action', 10)
        self._cubes_pub = self.create_publisher(PoseArray, '/nav/objects/cubes', 10)
        self._target_pub = self.create_publisher(PoseStamped, '/nav/target/cube', 10)

        
        self.create_subscription(String, '/nav/status', self.status_callback, 10)
        self.create_subscription(String, '/arm/result', self.arm_result_callback, 10)
        self.create_subscription(PointStamped, '/detection/objects/blue_cube', self.cube_callback, 10)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._static_loaded = False
        self.create_timer(0.5, self.try_load_static_frames_once)

        self._waiting_for_result = False

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
            elif not self.manual_goal and self._state == AutoState.APPROACH_OBJECT:
                if msg.data == 'REACHED':
                    self.get_logger().info('Approach goal reached. Triggering arm pickup.')
                    self.publish_arm_status('PICK_UP')
                    self._state = AutoState.WAIT_PICKUP_RESULT
                else:
                    self.get_logger().warn('Approach goal failed. Pickup command will not be sent.')
            elif not self.manual_goal and self._state == AutoState.RETURN_HOME:
                if msg.data == 'REACHED':
                    self.get_logger().info('Home reached. Triggering arm drop.')
                    self.publish_arm_status('DROP')
                    self._state = AutoState.WAIT_DROP_RESULT
                else:
                    self.get_logger().warn('Return-home goal failed. Drop command will not be sent.')

    # ----------------------------

    def arm_result_callback(self, msg: String):
        if self.manual_goal:
            return

        if self._state == AutoState.WAIT_PICKUP_RESULT:
            if msg.data == 'PICK_UP_SUCCESS':
                self.get_logger().info('Arm pickup succeeded. Returning home.')
                # Remove picked cube from list (best-effort) and clear current target
                if self._target_ is not None:
                    tx, ty = self._target_
                    self._cubes = [(x, y) for (x, y) in self._cubes if math.hypot(x - tx, y - ty) > self._merge_radius]
                self._target_ = None
                self.publish_topics()

                self._state = AutoState.RETURN_HOME
                self.publish_goal(self._home_x, self._home_y, self._home_yaw)
            elif msg.data in ('PICK_UP_FAIL_NO_OBJECT', 'PICK_UP_FAIL_NO_START'):
                self.get_logger().warn(f'Arm pickup failed: {msg.data}')
            else:
                self.get_logger().info(f'Arm result received while waiting for pickup: {msg.data}')
            return

        if self._state == AutoState.WAIT_DROP_RESULT:
            if msg.data == 'DROP_SUCCESS':
                self.get_logger().info('Drop succeeded.')

                self._detection_locked = False

                # If we still have cubes, go for the closest one; else go to SEARCH point
                if len(self._cubes) > 0:
                    robot_xy = self.get_robot_xy()
                    if robot_xy is not None:
                        rx, ry = robot_xy
                        best = min(self._cubes, key=lambda c: math.hypot(c[0] - rx, c[1] - ry))
                        self._target_ = best
                        self.publish_topics()

                        tx, ty = best
                        heading = math.atan2(ty - ry, tx - rx)
                        dist = math.hypot(tx - rx, ty - ry)
                        if dist > 1e-6:
                            if dist <= self._approach_distance:
                                ax, ay = rx, ry
                            else:
                                ax = tx - self._approach_distance * math.cos(heading)
                                ay = ty - self._approach_distance * math.sin(heading)
                            ayaw = math.atan2(ty - ay, tx - ax)

                            self._state = AutoState.APPROACH_OBJECT
                            self.publish_goal(ax, ay, ayaw)
                            return

                # No cubes known: return to SEARCH
                self._state = AutoState.SEARCH
                self.publish_goal(self._search_x, self._search_y, self._search_yaw)
                self.get_logger().info('State SEARCH: navigating to fixed search point while detection runs.')
                
            elif msg.data == 'DROP_FAIL_NO_OBJECT':
                self.get_logger().warn('Drop failed: DROP_FAIL_NO_OBJECT')
            else:
                self.get_logger().info(f'Arm result received while waiting for drop: {msg.data}')


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
            self._cubes.append((ox, oy))
            seeded += 1

        if seeded > 0:
            self.get_logger().info(f'Seeded {seeded} cubes from static TF frames ({self._object_frame_prefix}0..).')
            self.publish_topics()
        
        if (not self.manual_goal) and (self._state in (AutoState.IDLE, AutoState.INITIALIZATION, AutoState.SEARCH)) and len(self._cubes) > 0:
            # Kick off by selecting closest cube as target (same logic as cube_callback)
            robot_xy = self.get_robot_xy()
            if robot_xy is not None:
                rx, ry = robot_xy
                best = min(self._cubes, key=lambda c: math.hypot(c[0] - rx, c[1] - ry))
                self._target_ = best
                self.publish_topics()
                tx, ty = best
                heading = math.atan2(ty - ry, tx - rx)
                dist = math.hypot(tx - rx, ty - ry)
                if dist > 1e-6:
                    if dist <= self._approach_distance:
                        ax, ay = rx, ry
                    else:
                        ax = tx - self._approach_distance * math.cos(heading)
                        ay = ty - self._approach_distance * math.sin(heading)
                    ayaw = math.atan2(ty - ay, tx - ax)
                    self._state = AutoState.APPROACH_OBJECT
                    self.publish_goal(ax, ay, ayaw)

        self._static_loaded = True
    # ----------------------------



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
        if self.manual_goal:
            return

        # Accept detections in SEARCH, APPROACH_OBJECT, RETURN_HOME
        if self._state not in (AutoState.SEARCH, AutoState.APPROACH_OBJECT, AutoState.RETURN_HOME):
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
            self.get_logger().info(f'New cube added at x={ox:.2f}, y={oy:.2f} (total={len(self._cubes)})')

        # Always publish cube topics so planner can avoid them (even during RETURN_HOME)
        self.publish_topics()

        # During RETURN_HOME: do NOT switch goal (we're carrying)
        if self._state == AutoState.RETURN_HOME:
            return

        # Need robot pose to select closest target
        robot_xy = self.get_robot_xy()
        if robot_xy is None:
            self.get_logger().warn('Cube detected, but robot pose is unavailable. Ignoring goal update.')
            return
        rx, ry = robot_xy

        # Select closest cube as target (for now: always switch on new cube; thrashing prevention later)
        best = None
        best_d = float('inf')
        for (cx, cy) in self._cubes:
            d = math.hypot(cx - rx, cy - ry)
            if d < best_d:
                best_d = d
                best = (cx, cy)

        if best is None:
            return

        # If target unchanged (within merge radius), keep it
        if self._target_ is not None:
            if math.hypot(best[0] - self._target_[0], best[1] - self._target_[1]) <= self._merge_radius:
                # Still same target; no need to spam goals
                return

        # Switch target
        self._target_ = best
        self.publish_topics()  # publish updated target immediately

        tx, ty = best
        dx = tx - rx
        dy = ty - ry
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return

        heading_to_object = math.atan2(dy, dx)

        # Approach point with standoff distance
        if dist <= self._approach_distance:
            ax, ay = rx, ry
        else:
            ax = tx - self._approach_distance * math.cos(heading_to_object)
            ay = ty - self._approach_distance * math.sin(heading_to_object)

        ayaw = math.atan2(ty - ay, tx - ax)

        # Enter/keep approach state; trigger replanning via new goal
        self._state = AutoState.APPROACH_OBJECT
        self.get_logger().info(
            f'Target  cube at x={tx:.2f}, y={ty:.2f}. '
            f'Publishing approach goal x={ax:.2f}, y={ay:.2f}, yaw={ayaw:.2f}.'
        )
        self.publish_goal(ax, ay, ayaw)


    # ----------------------------

    def get_robot_xy(self):
        try:
            t = self._tf_buffer.lookup_transform(
                self._fixed_frame, self._base_frame, rclpy.time.Time())
        except Exception:
            return None
        return t.transform.translation.x, t.transform.translation.y

    # ----------------------------

    def publish_goal(self, gx, gy, gyaw=0.0):
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

    # ----------------------------

    def publish_arm_status(self, text: str):
        msg = String()
        msg.data = text
        self._arm_status_pub.publish(msg)
        self.get_logger().info(f'Arm status command sent: {text}')

    # ----------------------------

    def start_autonomous_sequence(self):
        self._state = AutoState.INITIALIZATION
        self.publish_arm_status('START')

        self._state = AutoState.SEARCH
        self.publish_goal(self._search_x, self._search_y, self._search_yaw)
        self.get_logger().info('State SEARCH: navigating to fixed search point while detection runs.')

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
