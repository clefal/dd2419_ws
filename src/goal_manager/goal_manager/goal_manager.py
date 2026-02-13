#!/usr/bin/env python3

import random
import math
import sys
import threading

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from tf_transformations import quaternion_from_euler
from tf2_ros import Buffer, TransformListener


class GoalManager(Node):

    def __init__(self):
        super().__init__('goal_manager')

        self.manual_goal = True

        # Workspace (odom frame)
        self._xmin = -1
        self._xmax = 1
        self._ymin = -1
        self._ymax = 1

        self._min_dist = 0.5

        self._fixed_frame = 'odom'
        self._base_frame = 'base_link'


        self._goal_pub = self.create_publisher(PoseStamped, '/nav/goal', 10)
        self.create_subscription(String, '/nav/status', self.status_callback, 10)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._waiting_for_result = False

        if self.manual_goal:
            thread = threading.Thread(target=self.manual_input_loop, daemon=True)
            thread.start()
        else:
            self.publish_new_goal()

    # ----------------------------

    def status_callback(self, msg: String):
        if not self._waiting_for_result:
            return

        if msg.data in ('REACHED', 'FAILED'):
            self._waiting_for_result = False
            if not self.manual_goal:
                self.publish_new_goal()

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

    def publish_new_goal(self):
        robot_xy = self.get_robot_xy()

        for _ in range(50):
            gx = random.uniform(self._xmin, self._xmax)
            gy = random.uniform(self._ymin, self._ymax)

            if robot_xy is None:
                break

            rx, ry = robot_xy
            if math.hypot(gx - rx, gy - ry) > self._min_dist:
                break

        self.publish_goal(gx, gy, 0.0)

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
                gyaw = float(parts[2])

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
