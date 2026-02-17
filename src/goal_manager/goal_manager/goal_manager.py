#!/usr/bin/env python3

import math
import sys
import threading
from enum import Enum

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, PointStamped
from std_msgs.msg import String
from tf_transformations import quaternion_from_euler
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
        self._latest_blue_cube = None
        self._blue_detection_locked = False
        self._approach_distance = 0.18

        self._search_x = 3.0
        self._search_y = 0.0
        self._search_yaw = 0.0
        self._home_x = 0.0
        self._home_y = 0.0
        self._home_yaw = 0.0

        self._fixed_frame = 'odom'
        self._base_frame = 'base_link'


        self._goal_pub = self.create_publisher(PoseStamped, '/nav/goal', 10)
        self._arm_status_pub = self.create_publisher(String, '/arm/action', 10)
        self.create_subscription(String, '/nav/status', self.status_callback, 10)
        self.create_subscription(String, '/arm/result', self.arm_result_callback, 10)
        self.create_subscription(PointStamped, '/detection/objects/blue_cube', self.blue_cube_callback, 10)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

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
                self._state = AutoState.RETURN_HOME
                self.publish_goal(self._home_x, self._home_y, self._home_yaw)
            elif msg.data in ('PICK_UP_FAIL_NO_OBJECT', 'PICK_UP_FAIL_NO_START'):
                self.get_logger().warn(f'Arm pickup failed: {msg.data}')
            else:
                self.get_logger().info(f'Arm result received while waiting for pickup: {msg.data}')
            return

        if self._state == AutoState.WAIT_DROP_RESULT:
            if msg.data == 'DROP_SUCCESS':
                self.get_logger().info('Drop succeeded. Entering IDLE state.')
                self._blue_detection_locked = False
                self._state = AutoState.IDLE
            elif msg.data == 'DROP_FAIL_NO_OBJECT':
                self.get_logger().warn('Drop failed: DROP_FAIL_NO_OBJECT')
            else:
                self.get_logger().info(f'Arm result received while waiting for drop: {msg.data}')

    # ----------------------------

    def blue_cube_callback(self, msg: PointStamped):
        if self.manual_goal:
            return

        if self._blue_detection_locked:
            return

        if self._state != AutoState.SEARCH:
            return

        robot_xy = self.get_robot_xy()
        if robot_xy is None:
            self.get_logger().warn('Blue cube detected, but robot pose is unavailable. Ignoring detection.')
            return

        self._latest_blue_cube = msg
        rx, ry = robot_xy
        ox = msg.point.x
        oy = msg.point.y

        dx = ox - rx
        dy = oy - ry
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            self.get_logger().warn('Blue cube detection is too close to robot pose; ignoring detection.')
            return

        heading_to_object = math.atan2(dy, dx)
        if dist <= self._approach_distance:
            ax = rx
            ay = ry
        else:
            ax = ox - self._approach_distance * math.cos(heading_to_object)
            ay = oy - self._approach_distance * math.sin(heading_to_object)

        ayaw = math.atan2(oy - ay, ox - ax)

        self._blue_detection_locked = True
        self._state = AutoState.APPROACH_OBJECT
        self.get_logger().info(
            f'Blue cube detected at x={ox:.2f}, y={oy:.2f}. '
            f'Preempting SEARCH goal with approach goal x={ax:.2f}, y={ay:.2f}, yaw={ayaw:.2f}.'
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
