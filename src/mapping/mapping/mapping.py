import math
import rclpy
import numpy as np

from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException

class Mapping(Node):
    def __init__(self):
        super().__init__('mapping')

        # Params
        self.declare_parameter("ocuppancy_grid_topic", "/map/occupancy_grid")
        self.declare_parameter("grid_resolution", 0.05) # m/cell
        self.declare_parameter("lidar_topic", "/lidar/scan")
        self.declare_parameter("scans_to_skip", 5)
        self.declare_parameter("is_turning_topic", "/nav/is_turning")
        self.declare_parameter("grid_size", 20) # m
        self.declare_parameter("grid_origin", [-5.0, -5.0]) # m


        # Config params
        self.scans_to_skip = self.get_parameter("scans_to_skip").value
        self.grid_resolution = self.get_parameter("grid_resolution").value
        self.range_min = 0.1
        self.range_max = 4.0

        # Occupancy grid publisher
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,    
        )
        ocuppancy_grid_topic = self.get_parameter("ocuppancy_grid_topic").value
        self.ocuppancy_grid_publisher = self.create_publisher(OccupancyGrid, ocuppancy_grid_topic, qos)

        # Lidar subscriber
        lidar_topic = self.get_parameter("lidar_topic").value
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            lidar_topic,
            self.lidar_callback,
            10)
        self.lidar_subscription  # prevent unused variable warning
        self.skipped_scans = 0

        # Is turning subscriber
        is_turning_topic = self.get_parameter("is_turning_topic").value
        self.is_turning_subscription = self.create_subscription(
            Bool,
            is_turning_topic,
            self.is_turning_callback,
            10)
        self.is_turning_subscription  # prevent unused variable warning
        self.is_turning = False

        # Grid
        self.grid_size = self.get_parameter("grid_size").value
        self.grid_origin = self.get_parameter("grid_origin").value
        self.grid = OcupancyGridData(self.grid_size, self.grid_resolution, self.grid_origin)

        # Publish init grid
        occupancy_grid_msg = OccupancyGrid()
        occupancy_grid_msg.header.stamp = self.get_clock().now().to_msg()
        occupancy_grid_msg.header.frame_id = "map"
        occupancy_grid_msg.info.resolution = self.grid.resolution
        occupancy_grid_msg.info.width = self.grid.width
        occupancy_grid_msg.info.height = self.grid.height
        occupancy_grid_msg.info.origin.position.x = self.grid.origin[0]
        occupancy_grid_msg.info.origin.position.y = self.grid.origin[1]
        occupancy_grid_msg.info.origin.position.z = 0
        occupancy_grid_msg.data = self.grid.get_data()

        self.ocuppancy_grid_publisher.publish(occupancy_grid_msg)
        

        self.get_logger().info("Mapping node started")

        # TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)



    def lidar_callback(self, msg: LaserScan):
        if self.is_turning:
            return
        
        if self.skipped_scans < self.scans_to_skip:
            self.skipped_scans += 1
            return

        self.skipped_scans = 0

        try:
            tf = self.tf_buffer.lookup_transform(
                "map",
                msg.header.frame_id,
                msg.header.stamp,
            )
        except TransformException as ex:
            self.get_logger().warn(f"TF lookup failed: {ex}")
            return

        x_robot, y_robot = tf.transform.translation.x, tf.transform.translation.y
        q = tf.transform.rotation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

        filtered_ranges = self.median_filter_scan(msg.ranges, kernel_size=25)

        for i in range(len(filtered_ranges)):
            r = filtered_ranges[i]

            if not math.isfinite(r) or r < self.range_min or r > self.range_max:
                continue



            
            r = msg.ranges[i]
            if not math.isfinite(r) or r < msg.range_min or r > msg.range_max:
                continue
            ang = msg.angle_min + i * msg.angle_increment
            x_scan = r * math.cos(ang)
            y_scan = r * math.sin(ang)
            # Rotate scan into map frame and translate by robot pose
            x = x_robot + (x_scan * math.cos(yaw) - y_scan * math.sin(yaw))
            y = y_robot + (x_scan * math.sin(yaw) + y_scan * math.cos(yaw))
            if np.isnan(x) or np.isnan(y):
                continue
            self.grid.update(x, y, 100)

        occupancy_grid_msg = OccupancyGrid()
        occupancy_grid_msg.header.stamp = msg.header.stamp
        occupancy_grid_msg.header.frame_id = "map"
        occupancy_grid_msg.info.resolution = self.grid.resolution
        occupancy_grid_msg.info.width = self.grid.width
        occupancy_grid_msg.info.height = self.grid.height
        occupancy_grid_msg.info.origin.position.x = self.grid.origin[0]
        occupancy_grid_msg.info.origin.position.y = self.grid.origin[1]
        occupancy_grid_msg.info.origin.position.z = 0
        occupancy_grid_msg.data = self.grid.get_data()

        self.ocuppancy_grid_publisher.publish(occupancy_grid_msg)

    def is_turning_callback(self, msg: Bool):
        self.is_turning = msg.data

    def median_filter_scan(self, ranges, kernel_size=3):
        ranges_np = np.array(ranges)

        # Replace invalid values with NaN for filtering
        invalid = ~np.isfinite(ranges_np)
        ranges_np[invalid] = np.nan

        filtered = np.copy(ranges_np)

        k = kernel_size // 2

        for i in range(k, len(ranges_np) - k):
            window = ranges_np[i - k:i + k + 1]
            window = window[np.isfinite(window)]  # remove NaNs

            if len(window) > 0:
                before_update = filtered[i]
                filtered[i] = np.median(window)
                self.get_logger().info(f"{before_update} -> {filtered[i]}")

        return filtered

def main():
    rclpy.init()
    node = Mapping()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


class OcupancyGridData:
    def __init__(self, size, resolution, origin):
        self.size = size # m
        self.resolution = resolution # m/cell
        self.width = int(size // resolution) 
        self.height = int(size // resolution)
        self.origin = origin # x, y in m
        self.grid = np.zeros((self.height, self.width), dtype=np.int8)

    def update(self, x, y, occupancy):
        if x < self.origin[0] or x > self.origin[0] + self.size or y < self.origin[1] or y > self.origin[1] + self.size:
            return
        
        x_index = int((x - self.origin[0]) // self.resolution)
        y_index = int((y - self.origin[1]) // self.resolution)

        if x_index < 0 or x_index >= self.width or y_index < 0 or y_index >= self.height:
            return

        self.grid[y_index, x_index] = occupancy

    def get_data(self):
        # Convert data to int8[] data
        return self.grid.reshape(-1).tolist()
