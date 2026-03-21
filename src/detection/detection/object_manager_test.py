#!/usr/bin/env python
import rclpy 
from rclpy.node import Node
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
        self.get_logger().info('Testing goals_available service asynchronously...')

        req = GoalsAvailable.Request()
        
        # Send the request asynchronously
        future = self.cli_goals_available.call_async(req)
        # Attach a callback function that will run ONLY when the response arrives.
        future.add_done_callback(self.goals_available_response_callback)
        

    def goals_available_response_callback(self, future):
        """This function is triggered automatically when the service responds."""
        try:
            # Extract the actual response from the Future object
            res = future.result()
            self.get_logger().info(f'Goals available service returned: {res.goals_available}')
        except Exception as e:
            # It's good practice to catch exceptions in case the service server crashed or failed
            self.get_logger().error(f'Service call failed: {e}')


    def test_get_closest_cube(self):
        self.get_logger().info(f'Testing get_closest_cube service')

        req = GetClosestCube.Request()
        req.robot_x = 4.0
        req.robot_y = 1.5
        future = self.cli_get_closest_cube.call_async(req)
        future.add_done_callback(self.test_get_closest_cube_callback)


    def test_get_closest_cube_callback(self, future):
        try:
            res = future.result()
            self.get_logger().info(f'closest cube to x=4.0 and y = 1.5 is {res.obj_id} at {res.obj_x}, {res.obj_y}')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')


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
