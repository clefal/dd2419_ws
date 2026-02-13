#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from robp_interfaces.msg import DutyCycles

from tf2_ros import Buffer, TransformListener
from tf_transformations import euler_from_quaternion


def wrap_angle(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class Controller(Node):

    def __init__(self):
        super().__init__('controller')

        # Publishers / subscribers
        self._cmd_pub = self.create_publisher(DutyCycles, '/phidgets/motor/duty_cycles', 10)
        self._status_pub = self.create_publisher(String, '/nav/status', 10)
        self.create_subscription(PoseStamped, '/nav/goal', self.goal_callback, 10)

        # TF to get robot pose (odom -> base_link)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._fixed_frame = 'odom'
        self._base_frame = 'base_link'

        # Current goal
        self._goal = None

        # Minimal tuning (duty cycles)
        self._max_duty = 0.3
        self._k_w = 0.25
        self._k_v = 0.6
        self._v_min = 0.08   # minimum duty that actually moves the robot


        # Tolerances
        self._xy_tol = 0.02 #0.1
        self._yaw_tol = 0.05 #0.25
        self._yaw_turn_thresh = 0.05 #0.35

        # Control loop
        self._timer = self.create_timer(0.1, self.control_tick)  # 10 Hz, encoders run at 20Hz

    def goal_callback(self, msg: PoseStamped):
        self._goal = msg
        self.publish_status('RUNNING')

        gx = msg.pose.position.x
        gy = msg.pose.position.y

        q = msg.pose.orientation
        gyaw = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]

        self.get_logger().info(f'Received new goal: x={gx:.3f}, y={gy:.3f}, yaw={gyaw:.3f} rad')


    def publish_status(self, s: str):
        msg = String()
        msg.data = s
        self._status_pub.publish(msg)

    def send_duty(self, left, right):
        m = DutyCycles()
        m.duty_cycle_left = float(clamp(left, -1.0, 1.0))
        m.duty_cycle_right = float(clamp(right, -1.0, 1.0))
        self._cmd_pub.publish(m)

    def stop(self):
        self.send_duty(0.0, 0.0)

    def get_pose_2d(self):
        try:
            t = self._tf_buffer.lookup_transform(self._fixed_frame, self._base_frame, rclpy.time.Time())
        except Exception:
            return None

        x = t.transform.translation.x
        y = t.transform.translation.y
        q = t.transform.rotation
        yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        return x, y, yaw

    def control_tick(self):
        if self._goal is None:
            return

        pose = self.get_pose_2d()
        if pose is None:
            self.stop()
            self.publish_status('FAILED')
            self._goal = None
            self.get_logger().warn('No TF pose available (odom->base_link).')
            return

        x, y, yaw = pose

        gx = self._goal.pose.position.x
        gy = self._goal.pose.position.y
        qg = self._goal.pose.orientation
        gyaw = euler_from_quaternion([qg.x, qg.y, qg.z, qg.w])[2]

        dx = gx - x
        dy = gy - y
        dist = math.hypot(dx, dy)

        heading = math.atan2(dy, dx)
        yaw_err_to_goal = wrap_angle(heading - yaw) #pointing angle
        yaw_err_final = wrap_angle(gyaw - yaw)      #requested goal angle

        # REACHED?
        if dist < self._xy_tol:
            if abs(yaw_err_final) < self._yaw_tol:
                self.stop()
                self.publish_status('REACHED')
                self._goal = None
                return
            # Final align
            w = clamp(self._k_w * yaw_err_final, -self._max_duty, self._max_duty)
            self.send_duty(-w, w)
            return

        # TURN first if needed
        if abs(yaw_err_to_goal) > self._yaw_turn_thresh:
            w = clamp(self._k_w * yaw_err_to_goal, -self._max_duty, self._max_duty)
            self.send_duty(-w, w)
            return

        # DRIVE (with heading correction)

        v = clamp(self._k_v * dist, 0.0, self._max_duty)
        w = clamp(self._k_w * yaw_err_to_goal, -self._max_duty, self._max_duty)
        v = max(v, self._v_min)
        
        left = v - w
        right = v + w
        self.send_duty(left, right)


def main():
    rclpy.init()
    node = Controller()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.stop()
        except Exception:
            pass
    rclpy.shutdown()


if __name__ == '__main__':
    main()
