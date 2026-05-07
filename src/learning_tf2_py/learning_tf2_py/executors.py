from example_interfaces.srv import AddTwoInts

import rclpy
from rclpy.node import Node

from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor

import time

class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')

        self.srv_1 = self.create_service(AddTwoInts, 'first_srv', self.first_callback)
        self.srv_2 = self.create_service(AddTwoInts, 'second_srv', self.second_callback)

    def first_callback(self, request, response):
        self.get_logger().info('Entered 1st callback')
        time.sleep(5)
        response.sum = request.a + request.b
        self.get_logger().info('Exited  1st callback')

        return response

    def second_callback(self, request, response):
        self.get_logger().info('Entered 2nd callback')
        response.sum = request.a + request.b
        self.get_logger().info('Exited  2nd callback')

        return response


def main():
    rclpy.init()
    node = MinimalService()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()