#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node

from tf2_ros import TransformBroadcaster
from tf_transformations import quaternion_from_euler

from geometry_msgs.msg import TransformStamped
from robp_interfaces.msg import Encoders
from sensor_msgs.msg import Imu
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

from tf_transformations import euler_from_quaternion


def wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class Odometry(Node):

    def __init__(self):
        super().__init__('odometry')

        # TF broadcaster
        self._tf_broadcaster = TransformBroadcaster(self)

        # Path publisher
        self._path_pub = self.create_publisher(Path, 'path', 10)
        self._path = Path()

        # Subscriptions
        self.create_subscription(
            Encoders, '/phidgets/motor/encoders', self.encoder_callback, 10
        )
        self.create_subscription(
            Imu, '/phidgets/imu/data_raw', self.imu_callback, 50
        )

        # -------------------------
        # State
        # -------------------------
        self._x = 0.0
        self._y = 0.0

        # Encoder yaw (pure wheel odom)
        self._yaw_enc = 0.0

        # Fused yaw
        self._yaw = 0.0

        # Gyro integration bookkeeping
        self._last_imu_t = None
        # TODO: Run imu data collection for some seconds to calculate bias
        self._gyro_bias = 0.0

        # To handle startup nicely
        self._have_encoders = False


        # Init 
        self._initial_yaw_imu = None

        # -------------------------
        # Complementary filter gains
        # -------------------------
        # Encoder correction gain (0..1). Smaller = trust IMU more.
        self._k = 0.0
        # -------------------------
        # Robot model constants
        # -------------------------
        self._ticks_per_rev = 48 * 64   # measured: 3200, not 3074
        self._wheel_radius = 0.04921
        self._base = 0.3075

    def imu_callback(self, msg: Imu):
        t = stamp_to_sec(msg.header.stamp)

        # Wait until we have encoder yaw to initialize nicely
        if not self._have_encoders:
            self._last_imu_t = t
            return

        if self._last_imu_t is None:
            self._last_imu_t = t
            return

        dt = t - self._last_imu_t
        if dt <= 0.0 or dt > 0.5:
            # Skip weird timing jumps (startup / clock issues)
            self._last_imu_t = t
            return

        self._last_imu_t = t

        # Gyro z (yaw rate)
        # omega_z = - msg.angular_velocity.z

        # Predict (integrate gyro)
        # self._yaw = wrap_angle(self._yaw + (omega_z - self._gyro_bias) * dt)

        if self._initial_yaw_imu is None:
            self._initial_yaw_imu = wrap_angle(euler_from_quaternion([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])[2])
        
        self._yaw = - wrap_angle(euler_from_quaternion([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])[2] - self._initial_yaw_imu)

        # Publish TF at IMU rate for smooth orientation
        # self.broadcast_transform(msg.header.stamp, self._x, self._y, self._yaw)


    def encoder_callback(self, msg: Encoders):
        # Ticks since last message
        delta_ticks_left = msg.delta_encoder_left
        delta_ticks_right = msg.delta_encoder_right

        meters_per_tick = (2.0 * math.pi * self._wheel_radius) / self._ticks_per_rev
        d_left = meters_per_tick * delta_ticks_left
        d_right = meters_per_tick * delta_ticks_right

        d_s = 0.5 * (d_left + d_right)
        d_theta = (d_right - d_left) / self._base

        # ---- Update yaw ----
        self._yaw_enc = wrap_angle(self._yaw_enc + d_theta)

        # ---- Correct fused yaw using encoder yaw ----
        err = wrap_angle(self._yaw_enc - self._yaw)
        self._yaw = wrap_angle(self._yaw + self._k * err)

        mid_yaw = self._yaw + 0.5 * d_theta
        self._x += d_s * math.cos(mid_yaw)
        self._y += d_s * math.sin(mid_yaw)

        # ---- Initialize fused yaw ----
        if not self._have_encoders:
            self._have_encoders = True
            self._yaw = self._yaw_enc
            self._last_imu_t = stamp_to_sec(msg.header.stamp)

        # Publish TF
        stamp = msg.header.stamp
        self.broadcast_transform(stamp, self._x, self._y, self._yaw)

        # Path at encoder rate
        self.publish_path(stamp, self._x, self._y, self._yaw)

    def broadcast_transform(self, stamp, x, y, yaw):
        print(f'Distance to origin: {math.sqrt(x * x + y * y)} meters')
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'

        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = 0.0

        q = quaternion_from_euler(0.0, 0.0, yaw)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self._tf_broadcaster.sendTransform(t)

    def publish_path(self, stamp, x, y, yaw):
        self._path.header.stamp = stamp
        self._path.header.frame_id = 'odom'

        pose = PoseStamped()
        pose.header = self._path.header

        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.01

        q = quaternion_from_euler(0.0, 0.0, yaw)
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]

        self._path.poses.append(pose)
        self._path_pub.publish(self._path)


def main():
    rclpy.init()
    node = Odometry()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == '__main__':
    main()