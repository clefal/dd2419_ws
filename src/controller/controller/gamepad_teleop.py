#!/usr/bin/env python3

from typing import Optional

import rclpy
from rclpy.node import Node

from robp_interfaces.msg import DutyCycles
from sensor_msgs.msg import Joy


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


class GamepadTeleop(Node):
    def __init__(self):
        super().__init__('gamepad_teleop')

        self.declare_parameter('forward_axis', 1)
        self.declare_parameter('turn_axis', 0)
        self.declare_parameter('deadman_button', 5)
        self.declare_parameter('stop_button', 1)
        self.declare_parameter('max_duty_cycle', 0.20)
        self.declare_parameter('boost_multiplier', 2.0)
        self.declare_parameter('boost_axis', 4)
        self.declare_parameter('boost_threshold', -0.5)
        self.declare_parameter('turn_scale', 0.75)
        self.declare_parameter('axis_deadzone', 0.08)
        self.declare_parameter('joy_timeout', 0.35)
        self.declare_parameter('publish_rate_hz', 20.0)

        self._cmd_pub = self.create_publisher(DutyCycles, '/phidgets/motor/duty_cycles', 10)
        self.create_subscription(Joy, '/joy', self.joy_callback, 10)

        self._latest_joy: Optional[Joy] = None
        self._latest_joy_time = None
        self._last_left = 0.0
        self._last_right = 0.0
        self._timed_out_logged = False

        publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        timer_period = 1.0 / max(publish_rate_hz, 1.0)
        self._timer = self.create_timer(timer_period, self.control_tick)

    def joy_callback(self, msg: Joy):
        self._latest_joy = msg
        self._latest_joy_time = self.get_clock().now()
        self._timed_out_logged = False

    def control_tick(self):
        if self._latest_joy is None or self._latest_joy_time is None:
            self.publish_stop_if_needed()
            return

        timeout_s = float(self.get_parameter('joy_timeout').value)
        age_s = (self.get_clock().now() - self._latest_joy_time).nanoseconds * 1e-9
        if age_s > timeout_s:
            if not self._timed_out_logged:
                self.get_logger().warn(f'Joystick timeout after {age_s:.2f} s, stopping motors')
                self._timed_out_logged = True
            self.publish_stop_if_needed()
            return

        joy = self._latest_joy
        if self.button_pressed(joy, int(self.get_parameter('stop_button').value)):
            self.publish_stop_if_needed()
            return

        if not self.button_pressed(joy, int(self.get_parameter('deadman_button').value)):
            self.publish_stop_if_needed()
            return

        forward = self.read_axis(joy, int(self.get_parameter('forward_axis').value))
        turn = self.read_axis(joy, int(self.get_parameter('turn_axis').value))

        deadzone = float(self.get_parameter('axis_deadzone').value)
        forward = 0.0 if abs(forward) < deadzone else forward
        turn = 0.0 if abs(turn) < deadzone else turn

        max_duty = float(self.get_parameter('max_duty_cycle').value)
        boost_multiplier = float(self.get_parameter('boost_multiplier').value)
        boost_axis = int(self.get_parameter('boost_axis').value)
        boost_threshold = float(self.get_parameter('boost_threshold').value)
        turn_scale = float(self.get_parameter('turn_scale').value)

        boost_active = self.read_axis(joy, boost_axis) <= boost_threshold
        if boost_active:
            max_duty *= boost_multiplier

        # Arcade drive gives the expected differential-drive behavior:
        # forward/back on the vertical axis, curvature from the horizontal axis.
        max_duty = clamp(max_duty, 0.0, 1.0)
        left = clamp(forward + turn_scale * turn, -1.0, 1.0) * max_duty
        right = clamp(forward - turn_scale * turn, -1.0, 1.0) * max_duty

        self.publish_duty(left, right)

    def read_axis(self, joy: Joy, axis_index: int) -> float:
        if axis_index < 0 or axis_index >= len(joy.axes):
            return 0.0
        return float(joy.axes[axis_index])

    def button_pressed(self, joy: Joy, button_index: int) -> bool:
        if button_index < 0 or button_index >= len(joy.buttons):
            return False
        return bool(joy.buttons[button_index])

    def publish_duty(self, left: float, right: float):
        if left == self._last_left and right == self._last_right:
            return

        msg = DutyCycles()
        msg.duty_cycle_left = float(left)
        msg.duty_cycle_right = float(right)
        self._cmd_pub.publish(msg)
        self._last_left = left
        self._last_right = right

    def publish_stop_if_needed(self):
        self.publish_duty(0.0, 0.0)


def main(args=None):
    rclpy.init(args=args)
    node = GamepadTeleop()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
