#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Imu
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
from tf_transformations import euler_from_quaternion, quaternion_from_euler


def wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


class Localization(Node):
    """Minimal localization fusion node.

    Input:
    - TF from odometry: odom -> base_link (encoder-based)
    - IMU gyro z: /phidgets/imu/data_raw

    Output:
    - TF: localization -> base_link (same x,y as odom TF, fused yaw)
    """

    def __init__(self):
        super().__init__('localization')

        self._odom_frame = self.declare_parameter('odom_frame', 'odom').value
        self._loc_frame = self.declare_parameter('loc_frame', 'localization').value
        self._base_frame = self.declare_parameter('base_frame', 'base_link').value
        self._imu_topic = self.declare_parameter('imu_topic', '/phidgets/imu/data_raw').value

        # 0.0 => trust odom yaw only, 1.0 => trust imu integration only
        self._alpha = float(self.declare_parameter('alpha', 0.8).value)
        self._alpha = max(0.0, min(1.0, self._alpha))

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._tf_broadcaster = TransformBroadcaster(self)

        self.create_subscription(Imu, self._imu_topic, self.imu_callback, 50)

        self._last_imu_t = None
        self._yaw_imu = 0.0
        self._have_imu = False
        self._imu_yaw_initialized = False

        self._timer = self.create_timer(0.02, self.publish_fused_tf)
        self.get_logger().info('localization started')

    def imu_callback(self, msg: Imu):
        t = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        gz = float(msg.angular_velocity.z)

        if self._last_imu_t is not None:
            dt = t - self._last_imu_t
            if 0.0 < dt < 0.2:
                self._yaw_imu = wrap_angle(self._yaw_imu + gz * dt)
                self._have_imu = True

        self._last_imu_t = t

    def publish_fused_tf(self):
        try:
            t = self._tf_buffer.lookup_transform(self._odom_frame, self._base_frame, rclpy.time.Time())
        except Exception:
            print(f'Could not lookup transform {self._odom_frame} -> {self._base_frame}')
            return

        x = t.transform.translation.x
        y = t.transform.translation.y
        q = t.transform.rotation
        yaw_odom = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]

        # Initialize imu yaw to odom yaw first time we have both
        if self._have_imu and not self._imu_yaw_initialized:
            self._yaw_imu = yaw_odom
            self._imu_yaw_initialized = True

        if self._have_imu:
            yaw_fused = wrap_angle(yaw_odom + self._alpha * wrap_angle(self._yaw_imu - yaw_odom))
        else:
            yaw_fused = yaw_odom

        out = TransformStamped()
        out.header.stamp = t.header.stamp
        out.header.frame_id = self._loc_frame
        out.child_frame_id = self._base_frame

        out.transform.translation.x = x
        out.transform.translation.y = y
        out.transform.translation.z = 0.0

        qf = quaternion_from_euler(0.0, 0.0, yaw_fused)
        out.transform.rotation.x = qf[0]
        out.transform.rotation.y = qf[1]
        out.transform.rotation.z = qf[2]
        out.transform.rotation.w = qf[3]

        self._tf_broadcaster.sendTransform(out)


def main():
    rclpy.init()
    node = Localization()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
