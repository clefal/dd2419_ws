# import rclpy
# from rclpy.node import Node
# from nav_msgs.msg import OccupancyGrid

# class Mapping(Node):
#     def __init__(self):
#         super().__init__('mapping')

#         # Params
#         self.declare_parameter("ocuppancy_grid_topic", "/map")

#         # Occupancy grid publisher
#         self.ocuppancy_grid_topic = self.get_parameter("ocuppancy_grid_topic").value
#         self.publisher = self.create_publisher(OccupancyGrid, self.ocuppancy_grid_topic, 10)

def main():
    pass
    # rclpy.init()
    # node = Mapping()
    # rclpy.spin(node)
    # node.destroy_node()
    # rclpy.shutdown()


if __name__ == '__main__':
    main()
