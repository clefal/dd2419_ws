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


class Threading(Node):

    def __init__(self):
        super().__init__('threading')

        # Initialize the transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Initialize the transform buffer
        self.tf_buffer = Buffer()

        # Initialize the transform listener
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def broadcast(self):
        # Publish three transforms

        # We create a transform between map -> robot
        t = TransformStamped()
        t.header.frame_id = 'map'
        t.child_frame_id = 'robot'

        t.header.stamp = rclpy.time.Time(seconds=10).to_msg()
        self.tf_broadcaster.sendTransform(t)

        t.header.stamp = rclpy.time.Time(seconds=20).to_msg()
        self.tf_broadcaster.sendTransform(t)

        t.header.stamp = rclpy.time.Time(seconds=30).to_msg()
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

    rclpy.init()
    node = Threading()
    node.broadcast()
    
    # rclpy.spin_once(node)
    # rclpy.spin_once(node)
    # rclpy.spin_once(node)
    # rclpy.spin_once(node)

    while rclpy.ok():
        time_point = float(input("Time point (s): "))
        node.listen(time_point)

    rclpy.shutdown()