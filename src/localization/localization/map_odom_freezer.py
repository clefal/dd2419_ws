#!/usr/bin/env python3

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node

from geometry_msgs.msg import TransformStamped
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformBroadcaster, TransformListener


class MapOdomFreezer(Node):
    def __init__(self):
        super().__init__('map_odom_freezer')

        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('source_odom_frame', 'odom_temp')
        self.declare_parameter('target_odom_frame', 'odom')
        self.declare_parameter('publish_rate_hz', 20.0)
        self.declare_parameter('lookup_timeout_s', 0.2)
        self.declare_parameter('service_name', '/localization/freeze_odom')

        self.map_frame = self.get_parameter('map_frame').value
        self.source_odom_frame = self.get_parameter('source_odom_frame').value
        self.target_odom_frame = self.get_parameter('target_odom_frame').value
        self.lookup_timeout_s = float(self.get_parameter('lookup_timeout_s').value)
        service_name = self.get_parameter('service_name').value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.committed_tf = None

        publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        publish_period = 1.0 / max(1.0, publish_rate_hz)
        self.publish_timer = self.create_timer(publish_period, self.publish_committed_tf)

        self.srv_freeze_odom = self.create_service(
            Trigger,
            service_name,
            self.freeze_odom_callback,
        )

        self.get_logger().info(
            f'MapOdomFreezer started. Service={service_name}, '
            f'source={self.map_frame}->{self.source_odom_frame}, '
            f'target={self.map_frame}->{self.target_odom_frame}, '
            f'publish_rate={1.0 / publish_period:.1f} Hz.'
        )

    def freeze_odom_callback(self, req, res):
        del req

        try:
            source_tf = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.source_odom_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=self.lookup_timeout_s),
            )
        except Exception as ex:
            res.success = False
            res.message = (
                f'Could not freeze {self.map_frame}->{self.target_odom_frame}: '
                f'lookup {self.map_frame}->{self.source_odom_frame} failed: {ex}'
            )
            self.get_logger().warn(res.message)
            return res

        committed = TransformStamped()
        committed.header.stamp = self.get_clock().now().to_msg()
        committed.header.frame_id = self.map_frame
        committed.child_frame_id = self.target_odom_frame
        committed.transform = source_tf.transform
        self.committed_tf = committed
        self.tf_broadcaster.sendTransform(committed)

        res.success = True
        res.message = (
            f'Froze {self.map_frame}->{self.source_odom_frame} as '
            f'{self.map_frame}->{self.target_odom_frame}.'
        )
        self.get_logger().info(res.message)
        return res

    def publish_committed_tf(self):
        if self.committed_tf is None:
            return

        self.committed_tf.header.stamp = self.get_clock().now().to_msg()
        self.tf_broadcaster.sendTransform(self.committed_tf)


def main(args=None):
    rclpy.init(args=args)
    node = MapOdomFreezer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
