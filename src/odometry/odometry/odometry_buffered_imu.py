#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from typing import List

from tf2_ros import TransformBroadcaster
from tf_transformations import quaternion_from_euler

from geometry_msgs.msg import TransformStamped
from robp_interfaces.msg import Encoders
from sensor_msgs.msg import Imu
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from tf_transformations import quaternion_matrix
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

        # self._yaw_file = open("yaw_log.txt", "a")

        # -------------------------
        # Parameters
        # -------------------------
        # Encoder correction gain is kept as a parameter for compatibility,
        # but it is not used in the buffered-IMU fusion below.
        self.declare_parameter('encoder_correction_gain', 0.0)
        self.declare_parameter('ticks_per_rev', 48 * 64)  # measured: 3200, not 3074
        self.declare_parameter('wheel_radius', 0.04921)
        self.declare_parameter('base', 0.3075)
        self.declare_parameter('fix_tilt', True)
        self.declare_parameter('gyro_bias_duration', 3.0)

        # IMU fusion/filtering parameters.
        # imu_yaw_weight = 0.0 -> only encoders, 1.0 -> only IMU yaw increment.
        # imu_lowpass_alpha near 1.0 -> less filtering, near 0.0 -> more smoothing.
        self.declare_parameter('imu_yaw_weight', 1.0)
        self.declare_parameter('imu_lowpass_alpha', 0.9)

        # -------------------------
        # Robot model constants
        # -------------------------
        self._ticks_per_rev = self.get_parameter('ticks_per_rev').value
        self._wheel_radius = self.get_parameter('wheel_radius').value
        self._base = self.get_parameter('base').value

        # -------------------------
        # Complementary / IMU params
        # -------------------------
        self._k = self.get_parameter('encoder_correction_gain').value
        self._fix_tilt = self.get_parameter('fix_tilt').value
        self._gyro_bias_duration = self.get_parameter('gyro_bias_duration').value
        self._imu_yaw_weight = self.get_parameter('imu_yaw_weight').value
        self._imu_lowpass_alpha = self.get_parameter('imu_lowpass_alpha').value

        # TF broadcaster
        self._tf_broadcaster = TransformBroadcaster(self)

        # Path publisher
        self._path_pub = self.create_publisher(Path, 'path', 10)
        self._path = Path()

        # Subscriptions
        self.create_subscription(
            Encoders, '/phidgets/motor/encoders', self.encoder_callback, 20
        )
        self.create_subscription(
            Imu, '/phidgets/imu/data_raw', self.imu_callback, 50
        )

        # -------------------------
        # State
        # -------------------------
        self._x = 0.0
        self._y = 0.0

        # Encoder yaw (pure wheel odom, useful for debugging)
        self._yaw_enc = 0.0

        # Fused yaw
        self._yaw = 0.0

        # Gyro buffering/filtering bookkeeping
        self._last_imu_t = None
        self._imu_buffer = []  # Stores tuples: (time_sec, filtered_omega_z)
        self._filtered_omega_z = 0.0
        self._last_encoder_t = None

        # Gyro bias bookkeeping
        self._gyro_bias_initialized = False
        self._gyro_bias_init_time = None
        self._gyro_bias_count = 0
        self._gyro_bias = 0.0
        if self._fix_tilt:
            self._gyro_bias_duration = 0.0

        # To handle startup nicely
        self._have_encoders = False
        self._last_encoder_left = None
        self._last_encoder_right = None

        # Init
        self._initial_yaw_imu = None

    def imu_callback(self, msg: Imu):
        t = stamp_to_sec(msg.header.stamp)

        # Calculate gyro bias
        if not self._gyro_bias_initialized:
            if self._gyro_bias_duration == 0.0:
                self._gyro_bias = 0.0
                self._gyro_bias_initialized = True
            elif self._gyro_bias_init_time is None:
                self._gyro_bias_init_time = t
                return
            elif t - self._gyro_bias_init_time < self._gyro_bias_duration:
                self._gyro_bias += msg.angular_velocity.z
                self._gyro_bias_count += 1
                return
            else:
                self._gyro_bias = self._gyro_bias / self._gyro_bias_count
                self._gyro_bias_initialized = True
                self.get_logger().info('--------------------------------')
                self.get_logger().info(
                    'IMU gyro bias calculated: %f with %d samples'
                    % (self._gyro_bias, self._gyro_bias_count)
                )
                self.get_logger().info('--------------------------------')

        # Wait until we have encoder yaw/time to initialize nicely.
        # The IMU callback does not update self._yaw anymore; it only buffers samples.
        if not self._have_encoders:
            self._last_imu_t = t
            return

        if self._last_imu_t is None:
            self._last_imu_t = t
            return

        dt = t - self._last_imu_t
        if dt <= 0.0:
            # Skip weird timing jumps (startup / clock issues)
            self._last_imu_t = t
            return
        elif dt > 0.5:
            self._last_imu_t = t
            self.get_logger().warn('Large dt between IMU messages: %f seconds' % dt)
            return

        self._last_imu_t = t

        if self._fix_tilt:
            # Yaw from IMU transforming angular velocity with robot orientation.
            q = [
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,
            ]

            R = quaternion_matrix(q)[:3, :3]

            # Angular velocity in body frame
            omega_body = [
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
            ]

            # Rotate gyro vector to world frame and use -Z as planar yaw rate.
            omega_z = -(R @ omega_body)[2]
        else:
            # Yaw from IMU angular velocity. Keep sign convention from previous code.
            omega_z = -msg.angular_velocity.z

        # Do not integrate yaw here. The encoder callback consumes the buffered
        # gyro measurements and integrates them over the encoder interval.
        omega_z = omega_z - self._gyro_bias

        # Low-pass filter the gyro yaw rate:
        # alpha near 1.0 = less filtering, alpha near 0.0 = more smoothing.
        self._filtered_omega_z = (
            self._imu_lowpass_alpha * omega_z
            + (1.0 - self._imu_lowpass_alpha) * self._filtered_omega_z
        )

        self._imu_buffer.append((t, self._filtered_omega_z))

        # Keep the buffer bounded in case encoder messages stop arriving.
        if len(self._imu_buffer) > 500:
            self._imu_buffer = self._imu_buffer[-500:]

        # # ---- Optional yaw from IMU orientation ----
        # if self._initial_yaw_imu is None:
        #     self._initial_yaw_imu = wrap_angle(
        #         euler_from_quaternion([
        #             msg.orientation.x,
        #             msg.orientation.y,
        #             msg.orientation.z,
        #             msg.orientation.w,
        #         ])[2]
        #     )
        # self._yaw = -wrap_angle(
        #     euler_from_quaternion([
        #         msg.orientation.x,
        #         msg.orientation.y,
        #         msg.orientation.z,
        #         msg.orientation.w,
        #     ])[2] - self._initial_yaw_imu
        # )

    def consume_imu_delta_yaw(self, t0: float, t1: float) -> float:
        """Integrate buffered IMU yaw-rate samples between two encoder timestamps.

        Returns the yaw change measured by the IMU during [t0, t1]. Old samples
        are discarded once consumed so each IMU sample is used only once.
        """
        if t0 is None or t1 is None or t1 <= t0:
            return 0.0

        # Split buffer into samples to consume and samples to keep.
        samples = []
        keep = []
        for t, omega_z in self._imu_buffer:
            if t0 <= t <= t1:
                samples.append((t, omega_z))
            elif t > t1:
                keep.append((t, omega_z))

        self._imu_buffer = keep

        if len(samples) < 2:
            return 0.0

        delta_yaw = 0.0
        for i in range(1, len(samples)):
            ta, wa = samples[i - 1]
            tb, wb = samples[i]
            dt = tb - ta

            # Reject weird timestamp jumps. Normal IMU dt should be much smaller.
            if 0.0 < dt < 0.1:
                # Trapezoidal integration is less noisy than rectangular integration.
                delta_yaw += 0.5 * (wa + wb) * dt

        return delta_yaw

    def encoder_callback(self, msg: Encoders):
        encoder_left = msg.encoder_left
        encoder_right = msg.encoder_right

        # First encoder message: store absolute counts, initialize timing,
        # publish initial pose, but do not move yet.
        if self._last_encoder_left is None or self._last_encoder_right is None:
            self._last_encoder_left = encoder_left
            self._last_encoder_right = encoder_right
            self._last_encoder_t = stamp_to_sec(msg.header.stamp)

            if not self._have_encoders:
                self._have_encoders = True
                self._last_imu_t = self._last_encoder_t

            self.broadcast_transform(msg.header.stamp, self._x, self._y, self._yaw, True)
            self.publish_path(msg.header.stamp, self._x, self._y, self._yaw)
            return

        # Ticks since last encoder message, calculated from absolute tick counts.
        delta_ticks_left = encoder_left - self._last_encoder_left
        delta_ticks_right = encoder_right - self._last_encoder_right
        self._last_encoder_left = encoder_left
        self._last_encoder_right = encoder_right

        meters_per_tick = (2.0 * math.pi * self._wheel_radius) / self._ticks_per_rev
        d_left = meters_per_tick * delta_ticks_left
        d_right = meters_per_tick * delta_ticks_right

        d_s = 0.5 * (d_left + d_right)
        d_theta = (d_right - d_left) / self._base

        stamp = msg.header.stamp
        encoder_t = stamp_to_sec(stamp)
        imu_dtheta = self.consume_imu_delta_yaw(self._last_encoder_t, encoder_t)
        self._last_encoder_t = encoder_t

        # Encoder-only yaw for debugging/comparison.
        self._yaw_enc = wrap_angle(self._yaw_enc + d_theta)

        # Fuse IMU and encoder yaw increments.
        # imu_yaw_weight = 0.0 -> only encoders
        # imu_yaw_weight = 1.0 -> only IMU
        if imu_dtheta != 0.0:
            dtheta_fused = (
                self._imu_yaw_weight * imu_dtheta
                + (1.0 - self._imu_yaw_weight) * d_theta
            )
        else:
            # If no IMU samples arrived in this interval, fall back to encoders.
            dtheta_fused = d_theta

        old_yaw = self._yaw
        self._yaw = wrap_angle(self._yaw + dtheta_fused)

        # Use the same fused rotation for the midpoint heading used in position update.
        mid_yaw = wrap_angle(old_yaw + 0.5 * dtheta_fused)
        self._x += d_s * math.cos(mid_yaw)
        self._y += d_s * math.sin(mid_yaw)

        # Publish TF
        self.broadcast_transform(stamp, self._x, self._y, self._yaw, True)

        # Path at encoder rate
        self.publish_path(stamp, self._x, self._y, self._yaw)

    def broadcast_transform(self, stamp, x, y, yaw, temp=False):
        # print(f'Distance to origin: {math.sqrt(x * x + y * y)} meters')
        tfs: List[TransformStamped] = []
        q = quaternion_from_euler(0.0, 0.0, yaw)

        # Temporary odom frame
        if temp:
            t_temp = TransformStamped()
            t_temp.header.stamp = stamp
            t_temp.header.frame_id = 'odom_temp'
            t_temp.child_frame_id = 'base_link_temp'

            t_temp.transform.translation.x = x
            t_temp.transform.translation.y = y
            t_temp.transform.translation.z = 0.0

            t_temp.transform.rotation.x = q[0]
            t_temp.transform.rotation.y = q[1]
            t_temp.transform.rotation.z = q[2]
            t_temp.transform.rotation.w = q[3]
            tfs.append(t_temp)

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'

        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = 0.0

        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        tfs.append(t)

        self._tf_broadcaster.sendTransform(tfs)

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
        # node._yaw_file.close()
        pass
    rclpy.shutdown()


if __name__ == '__main__':
    main()
