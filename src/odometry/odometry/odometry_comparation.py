#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node

from tf2_ros import TransformBroadcaster
from tf_transformations import (
    quaternion_from_euler,
    euler_from_quaternion
)

from geometry_msgs.msg import TransformStamped
from robp_interfaces.msg import Encoders
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Imu
from rclpy.qos import qos_profile_sensor_data


def wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


class Odometry(Node):

    def __init__(self):
        super().__init__('odometry')

        self._tf_broadcaster = TransformBroadcaster(self)

        # Publishers
        self._path_pub = self.create_publisher(Path, 'path', 10)
        self._path_imu_pub = self.create_publisher(Path, 'path/imu', 10)

        self._path = Path()
        self._path_imu = Path()

        # Subscriptions
        self.create_subscription(
            Encoders,
            'phidgets/motor/encoders',
            self.encoder_callback,
            10
        )

        self.create_subscription(
            Imu,
            'phidgets/imu/data_raw',
            self.imu_callback,
            qos_profile_sensor_data
        )

        # --------------------------
        # Pure encoder state
        # --------------------------
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0

        # --------------------------
        # Hybrid state (encoder + IMU yaw)
        # --------------------------
        self._x_hybrid = 0.0
        self._y_hybrid = 0.0
        self._yaw_imu = 0.0
        self._yaw_imu_rel = 0.0
        self._imu_yaw0 = None

        # Drive constants
        self._ticks_per_rev = 48 * 64
        self._wheel_radius = 0.04921
        self._base = 0.31

    # ==========================================================
    # IMU CALLBACK (only yaw)
    # ==========================================================
    def imu_callback(self, msg: Imu):

        q = msg.orientation
        quat = [q.x, q.y, q.z, q.w]
        _, _, yaw = euler_from_quaternion(quat)

        self._yaw_imu = wrap_angle(yaw)

        # Save initial yaw to remove constant offset
        if self._imu_yaw0 is None:
            self._imu_yaw0 = self._yaw_imu

        # Relative yaw aligned to encoder initial heading
        self._yaw_imu_rel = wrap_angle(-(self._yaw_imu - self._imu_yaw0))

    # ==========================================================
    # ENCODER CALLBACK
    # ==========================================================
    def encoder_callback(self, msg: Encoders):

        delta_ticks_left = msg.delta_encoder_left
        delta_ticks_right = msg.delta_encoder_right

        meters_per_tick = (
            2.0 * math.pi * self._wheel_radius
        ) / self._ticks_per_rev

        d_left = meters_per_tick * delta_ticks_left
        d_right = meters_per_tick * delta_ticks_right

        d_s = 0.5 * (d_left + d_right)
        d_theta = (d_right - d_left) / self._base

        # ------------------------------------------------------
        # 1️⃣ PURE ENCODER
        # ------------------------------------------------------
        mid_yaw = self._yaw + 0.5 * d_theta
        self._x += d_s * math.cos(mid_yaw)
        self._y += d_s * math.sin(mid_yaw)
        self._yaw = wrap_angle(self._yaw + d_theta)

        # ------------------------------------------------------
        # 2️⃣ HYBRID (encoder position + IMU yaw)
        # ------------------------------------------------------
        self._x_hybrid += d_s * math.cos(self._yaw_imu_rel)
        self._y_hybrid += d_s * math.sin(self._yaw_imu_rel)

        stamp = msg.header.stamp

        # Publish encoder result
        self.broadcast_transform(
            stamp,
            self._x,
            self._y,
            self._yaw,
            'base_link'
        )

        self.publish_path(
            stamp,
            self._x,
            self._y,
            self._yaw,
            self._path,
            self._path_pub
        )

        # Publish hybrid result
        self.broadcast_transform(
            stamp,
            self._x_hybrid,
            self._y_hybrid,
            self._yaw_imu_rel,
            'base_link_imu'
        )

        self.publish_path(
            stamp,
            self._x_hybrid,
            self._y_hybrid,
            self._yaw_imu_rel,
            self._path_imu,
            self._path_imu_pub
        )

    # ==========================================================
    # TF BROADCAST
    # ==========================================================
    def broadcast_transform(self, stamp, x, y, yaw, child_frame):

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'odom'
        t.child_frame_id = child_frame

        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = 0.0

        q = quaternion_from_euler(0.0, 0.0, yaw)

        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self._tf_broadcaster.sendTransform(t)

    # ==========================================================
    # PATH PUBLISH
    # ==========================================================
    def publish_path(self, stamp, x, y, yaw, path, publisher):

        path.header.stamp = stamp
        path.header.frame_id = 'odom'

        pose = PoseStamped()
        pose.header = path.header

        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.01

        q = quaternion_from_euler(0.0, 0.0, yaw)
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]

        path.poses.append(pose)

        # Limit path size to avoid RViz overflow
        if len(path.poses) > 2000:
            path.poses.pop(0)

        publisher.publish(path)


def main():
    rclpy.init()
    node = Odometry()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()