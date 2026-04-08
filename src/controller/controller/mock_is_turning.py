#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool
from robp_interfaces.msg import Encoders


class MockIsTurning(Node):
    def __init__(self):
        super().__init__('mock_is_turning')

        # Parameters
        self.declare_parameter('omega_threshold', 0.5)   # rad/s
        self.declare_parameter('wheel_radius', 0.04921)     # m
        self.declare_parameter('wheel_base', 0.3075)       # m
        self.declare_parameter('ticks_per_rev', 48 * 64)   # encoder ticks / wheel revolution
        self.declare_parameter('use_header_stamp', True)

        # Publisher
        self.pub = self.create_publisher(Bool, '/nav/is_turning', 10)

        # Subscriber
        self.create_subscription(
            Encoders,
            '/phidgets/motor/encoders',
            self.encoder_callback,
            10
        )

        self.prev_time = None
        self.prev_state = None

        self.get_logger().info('mock_is_turning started')

    def _stamp_to_sec(self, stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def encoder_callback(self, msg: Encoders):
        # IMPORTANT:
        # Adjust these field names if your Encoders.msg uses different names.
        delta_left = float(msg.delta_encoder_left)
        delta_right = float(msg.delta_encoder_right)

        use_header_stamp = bool(self.get_parameter('use_header_stamp').value)

        if use_header_stamp and hasattr(msg, 'header'):
            current_time = self._stamp_to_sec(msg.header.stamp)
        else:
            current_time = self.get_clock().now().nanoseconds * 1e-9

        if self.prev_time is None:
            self.prev_time = current_time
            return

        dt = current_time - self.prev_time
        self.prev_time = current_time

        if dt <= 1e-6:
            self.get_logger().warn('dt too small, skipping encoder sample')
            return

        wheel_radius = float(self.get_parameter('wheel_radius').value)
        wheel_base = float(self.get_parameter('wheel_base').value)
        ticks_per_rev = float(self.get_parameter('ticks_per_rev').value)
        omega_threshold = float(self.get_parameter('omega_threshold').value)

        # Convert encoder deltas to traveled distance for each wheel
        meters_per_tick = (2.0 * math.pi * wheel_radius) / ticks_per_rev
        d_left = delta_left * meters_per_tick
        d_right = delta_right * meters_per_tick

        # Wheel linear velocities
        v_left = d_left / dt
        v_right = d_right / dt

        # Differential drive angular velocity
        omega = (v_right - v_left) / wheel_base

        is_turning = abs(omega) > omega_threshold
        if self.prev_state is not None:
            if self.prev_state != is_turning:
                self.get_logger().info(f"Turning state changed from {self.prev_state} to {is_turning}")
        self.prev_state = is_turning

        out_msg = Bool()
        out_msg.data = is_turning
        self.get_logger().info(f"is_turning: {is_turning}")
        self.pub.publish(out_msg)

        self.get_logger().debug(
            f"dt={dt:.4f}, dl={delta_left:.2f}, dr={delta_right:.2f}, "
            f"vl={v_left:.3f}, vr={v_right:.3f}, omega={omega:.3f}, "
            f"threshold={omega_threshold:.3f}, turning={is_turning}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = MockIsTurning()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()