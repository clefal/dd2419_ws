import math
import sys

from geometry_msgs.msg import TransformStamped

import numpy as np

import rclpy
import rclpy.duration
from rclpy.node import Node

import rclpy.time
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros import TransformBroadcaster
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


def quaternion_from_euler(ai, aj, ak):
    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    q = np.empty((4, ))
    q[0] = cj*sc - sj*cs
    q[1] = cj*ss + sj*cc
    q[2] = cj*cs - sj*sc
    q[3] = cj*cc + sj*ss

    return q


class Interpolation(Node):

    def __init__(self, static_transforms):
        super().__init__('interpolation')

        # Initialize the transform broadcaster
        if static_transforms:
            self.tf_broadcaster = StaticTransformBroadcaster(self)
        else:
            self.tf_broadcaster = TransformBroadcaster(self)

        # Initialize the transform buffer
        self.tf_buffer = Buffer()

        # Initialize the transform listener
        self.tf_listener = TransformListener(
            self.tf_buffer, self, spin_thread=True)

    def broadcast(self):
        # We create a transform between map -> robot
        t = TransformStamped()
        t.header.frame_id = 'map'
        t.child_frame_id = 'robot'

        # We publish it first with a timestamp of 10 seconds.
        # Typicall, you would not do this. You would either use the current time
        # or the timestamp of another message. However, in this case we want to
        # control the time so we can know what time points we can interpolate between.
        t.header.stamp = rclpy.time.Time(seconds=10).to_msg()

        # Send the first transformation
        self.tf_broadcaster.sendTransform(t)

        # We change the time to 20 seconds
        t.header.stamp = rclpy.time.Time(seconds=20).to_msg()

        # We also change the translation
        t.transform.translation.x = 1.0
        t.transform.translation.y = 2.0
        t.transform.translation.z = 3.0

        # Send the second transformation
        self.tf_broadcaster.sendTransform(t)

        # We change the time to 30 seconds
        t.header.stamp = rclpy.time.Time(seconds=30).to_msg()

        # We also change the rotation
        q = quaternion_from_euler(0, 0, math.radians(90))
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        # Send the third transformation
        self.tf_broadcaster.sendTransform(t)

    def listen(self, time_point):
        # Store frame names in variables that will be used to
        # compute transformations
        to_frame_rel = 'map'
        from_frame_rel = 'robot'
        time = rclpy.time.Time(seconds=time_point)

        # Look up the transformation between the frames
        try:
            t = self.tf_buffer.lookup_transform(
                to_frame_rel,
                from_frame_rel,
                time)
        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform {to_frame_rel} to {from_frame_rel}: {ex}')
            return

        print(f'Got transformation from {
              t.header.frame_id} to {t.child_frame_id}')
        print(f'Translation: x={t.transform.translation.x} y={
              t.transform.translation.y} z={t.transform.translation.z}')
        print(f'Rotation:    x={t.transform.rotation.x} y={t.transform.rotation.y} z={
              t.transform.rotation.z} w={t.transform.rotation.w}')


def main():
    logger = rclpy.logging.get_logger('logger')

    # obtain parameters from command line arguments
    if len(sys.argv) != 2:
        logger.info('Invalid number of parameters. Usage: \n'
                    '$ ros2 run learning_tf2_py interpolation static/dynamic')
        sys.exit(1)

    if sys.argv[1] != 'static' and sys.argv[1] != 'dynamic':
        logger.info('The parameter needs to be either "static" or "dynamic"')
        sys.exit(2)

    rclpy.init()
    if sys.argv[1] == 'static':
        node = Interpolation(True)
    else:
        node = Interpolation(False)

    node.broadcast()
    while rclpy.ok():
        time_point = float(input("Time point (s): "))
        node.listen(time_point)

    rclpy.shutdown()