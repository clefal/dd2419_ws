#!/usr/bin/env python3

import math

from robp_interfaces import msg
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
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

        #self._yaw_file = open("yaw_log.txt", "a")

        # -------------------------
        # Parameters
        # -------------------------
        self.declare_parameter('encoder_correction_gain', 0) # Encoder correction gain (0..1). Smaller = trust IMU more.
        self.declare_parameter('ticks_per_rev', 48 * 64) # measured: 3200, not 3074
        self.declare_parameter('wheel_radius', 0.04921)
        self.declare_parameter('base', 0.3075)
        self.declare_parameter('fix_tilt', True)
        self.declare_parameter('gyro_bias_duration', 3.0)

        # -------------------------
        # Robot model constants
        # -------------------------
        self._ticks_per_rev = self.get_parameter('ticks_per_rev').value 
        self._wheel_radius = self.get_parameter('wheel_radius').value
        self._base = self.get_parameter('base').value
        # -------------------------
        # Complementary params
        # -------------------------
        self._k = self.get_parameter('encoder_correction_gain').value
        self._fix_tilt = self.get_parameter('fix_tilt').value
        self._gyro_bias_duration = self.get_parameter('gyro_bias_duration').value


        # TF broadcaster
        self._tf_broadcaster = TransformBroadcaster(self)

        # Path publisher
        self._path_pub = self.create_publisher(Path, 'path', 10)
        self._path = Path()

        self.mut_ex_callback_group = MutuallyExclusiveCallbackGroup()
        # Subscriptions
        self.create_subscription(
            Encoders, '/phidgets/motor/encoders', self.encoder_callback, 20, callback_group=self.mut_ex_callback_group
        )
        self.create_subscription(
            Imu, '/phidgets/imu/data_raw', self.imu_callback, 50, callback_group=self.mut_ex_callback_group
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

        # -------------------------
        # Complementary filter gains
        # -------------------------
        # Encoder correction gain (0..1). Smaller = trust IMU more.

        # -------------------------
        # Robot model constants
        # -------------------------
        self._ticks_per_rev = self.get_parameter("ticks_per_rev").value   # measured: 3200, not 3074
        self._wheel_radius = self.get_parameter("wheel_radius").value
        self._base = self.get_parameter("base").value

    def imu_callback(self, msg: Imu):

        now = self.get_clock().now()
        msg_time = rclpy.time.Time.from_msg(msg.header.stamp)
        lag = (now - msg_time).nanoseconds * 1e-9

        if lag > 0.1:
            self.get_logger().warn(f"IMU callback lag: {lag:.3f} s")

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
                self.get_logger().info('IMU gyro bias calculated: %f with %d samples' % (self._gyro_bias, self._gyro_bias_count))
                self.get_logger().info('--------------------------------')
        

        
        # Wait until we have encoder yaw to initialize nicely
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
            # Yaw from IMU transforming the angular velocity with the orientation of the robot
            q = [msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w]

            R = quaternion_matrix(q)[:3, :3]

            # Angular velocity in body frame
            omega_body = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]

            # Rotate gyro vector to world frame and use -Z as planar yaw rate
            omega_z = -(R @ omega_body)[2]
        else:
            # Yaw from IMU angular velocity
            # Gyro z (yaw rate)
            omega_z = - msg.angular_velocity.z
        
        # Predict (integrate gyro)
        self._yaw = wrap_angle(self._yaw + (omega_z - self._gyro_bias) * dt)
        # self._yaw_file.write(f"{dt}: {self._yaw}\n")
        # self._yaw_file.flush()  # ensures it's written immediately





        # # ---- Yaw from IMU orientation ----
        # if self._initial_yaw_imu is None:
        #     self._initial_yaw_imu = wrap_angle(euler_from_quaternion([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])[2])
        
        # self._yaw = - wrap_angle(euler_from_quaternion([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])[2] - self._initial_yaw_imu)
        # # ----------------------------------

        # Publish both odom trees at IMU rate so scan-timestamped TF lookups do not
        # outrun the latest encoder-stamped odom_temp sample between encoder updates.
        # self.broadcast_transform(msg.header.stamp, self._x, self._y, self._yaw, False)


    def encoder_callback(self, msg: Encoders):

        now = self.get_clock().now()
        msg_time = rclpy.time.Time.from_msg(msg.header.stamp)
        lag = (now - msg_time).nanoseconds * 1e-9

        if lag > 0.1:
            self.get_logger().warn(f"Encoder callback lag: {lag:.3f} s")

        encoder_left = msg.encoder_left
        encoder_right = msg.encoder_right

        if self._last_encoder_left is None or self._last_encoder_right is None:
            self._last_encoder_left = encoder_left
            self._last_encoder_right = encoder_right

            if not self._have_encoders:
                self._have_encoders = True
                self._last_imu_t = stamp_to_sec(msg.header.stamp)

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
            self._last_imu_t = stamp_to_sec(msg.header.stamp)

        # Publish TF
        stamp = msg.header.stamp
        self.broadcast_transform(stamp, self._x, self._y, self._yaw, True)

        # Path at encoder rate
        self.publish_path(stamp, self._x, self._y, self._yaw)

    def broadcast_transform(self, stamp, x, y, yaw, temp=False):
        #print(f'Distance to origin: {math.sqrt(x * x + y * y)} meters')
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

    # ex = MultiThreadedExecutor()
    # ex.add_node(node)

    try:
        rclpy.spin(node)
        # ex.spin()
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()


if __name__ == '__main__':
    main()
