#!/usr/bin/env python
import rclpy 
import math
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, TransformStamped
from tf_transformations import quaternion_from_euler, euler_from_quaternion
from tf2_ros import Buffer, TransformListener, TransformBroadcaster, StaticTransformBroadcaster
from robp_interfaces.srv import GoalsAvailable, GetClosestCube

class ObjectManagerTest(Node):

    def __init__(self):
        super().__init__('object_manager_test')
        self.get_logger().info('Detection Manager Test node started.')
        
        self.cli_goals_available = self.create_client(GoalsAvailable, 'object_manager/goals_available')
        while not self.cli_goals_available.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('goals_available service not available, waiting again...')

        self.cli_get_closest_cube = self.create_client(GetClosestCube, 'object_manager/get_closest_cube')
        while not self.cli_get_closest_cube.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('get_closest_cube service not available, waiting again...')
            

        self.create_timer(1, self.test_goals_available)
        self.create_timer(1, self.test_get_closest_cube)

    
    def test_goals_available(self):
        self.get_logger().info(f'Testing goals_available service')

        req = GoalsAvailable.Request()
        res = self.cli_goals_available.call(req)   #maybe this would be better if we call it asyncronous, now we block this node 
        
        self.get_logger().info(f'Goals available service returned: {res.goals_available}')


    def test_get_closest_cube(self):
        self.get_logger().info(f'Testing get_closest_cube service')

        req = GetClosestCube.Request()
        req.robot_x = 4.0
        req.robot_y = 1.5
        res = self.cli_get_closest_cube.call(req)
        self.get_logger().info(f'closest cube to {req.robot_x}, {req.robot_y} is {res.obj_id} at {res.obj_x}, {res.obj_y}')


 
def main():
    rclpy.init()
    node = ObjectManagerTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()




