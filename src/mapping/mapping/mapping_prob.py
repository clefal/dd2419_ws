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
from geometry_msgs.msg import PolygonStamped

class Mapping(Node):
    def __init__(self):
        super().__init__('mapping')

        # Params
        self.declare_parameter("ocuppancy_grid_topic", "/map/occupancy_grid")
        self.declare_parameter("grid_resolution", 0.02) # m/cell
        self.declare_parameter("lidar_topic", "/lidar/scan")
        self.declare_parameter("scans_to_skip", 2)
        self.declare_parameter("is_turning_topic", "/nav/is_turning")
        self.declare_parameter("workspace_polygon_topic", "/workspace")
        self.declare_parameter("grid_size", 15) # m
        self.declare_parameter("grid_origin", [-3.0, -3.0]) # m


        # ------------------- CONFIG PARAMS ---------------
        # Grid params
        self.grid_size = self.get_parameter("grid_size").value
        self.grid_origin = self.get_parameter("grid_origin").value
        self.grid_resolution = self.get_parameter("grid_resolution").value

        # Lidar params
        self.scans_to_skip = self.get_parameter("scans_to_skip").value
        self.range_min = 0.1
        self.range_max = 5.0
        self.range_max_free_update = 3.0

        # Filter params
        self.median_filter_kernel_size = 5

        # Log-odds params
        self.log_odds_increse_occ = 0.85
        self.log_odds_decrease_free = -0.4
        self.log_odds_min = -5
        self.log_odds_max = 5

        
        # --------------------------------------------------

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
        self.grid = OcupancyGridData(self.grid_size, self.grid_resolution, self.grid_origin, self.log_odds_increse_occ, self.log_odds_decrease_free, self.log_odds_min, self.log_odds_max)

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

        # Workspace polygon subscriber
        workspace_polygon_topic = self.get_parameter("workspace_polygon_topic").value
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.workspace_polygon_subscription = self.create_subscription(
            PolygonStamped,
            workspace_polygon_topic,
            self.workspace_polygon_callback,
            qos
        )
        self.polygon = []
        

        self.get_logger().info("Mapping node started")

        # TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

    def workspace_polygon_callback(self, msg: PolygonStamped):
        self.polygon = np.array(
            [[pt.x, pt.y] for pt in msg.polygon.points],
            dtype=float
        )
        self.grid.set_workspace_polygon(self.polygon)
        self.grid.initialize_outside_polygon_as_occupied()

    def lidar_callback(self, msg: LaserScan):
        if self.is_turning:
            return
        
        if self.skipped_scans < self.scans_to_skip:
            self.skipped_scans += 1
            return

        self.skipped_scans = 0

        # Wait for the transform asynchronously
        try:
            tf_future = self.tf_buffer.wait_for_transform_async(
                "map",
                msg.header.frame_id,
                msg.header.stamp,
            )
            # # Spin until transform found or `timeout_sec` seconds has passed
            rclpy.spin_until_future_complete(self, tf_future, timeout_sec=2)
        except TransformException as ex:
            self.get_logger().warn(f"Async TF lookup failed: {ex}")
            return
        try:
            tf = self.tf_buffer.lookup_transform(
                "map",
                msg.header.frame_id,
                msg.header.stamp,
                timeout=rclpy.time.Duration(seconds=1)
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

        #self.get_logger().info(f"X: {x_robot}, Y: {y_robot}")

        filtered_ranges = self.median_filter_scan(msg.ranges, kernel_size=self.median_filter_kernel_size)
        last_cell_occupied = True

        for i in range(len(filtered_ranges)):
            r = filtered_ranges[i]
            last_cell_occupied = True

            if not math.isfinite(r) or r < self.range_min:
                continue
            if r > self.range_max:
                r = self.range_max_free_update
                last_cell_occupied = False
            ang = msg.angle_min + i * msg.angle_increment
            x_scan = r * math.cos(ang)
            y_scan = r * math.sin(ang)
            # Rotate scan into map frame and translate by robot pose
            x = x_robot + (x_scan * math.cos(yaw) - y_scan * math.sin(yaw))
            y = y_robot + (x_scan * math.sin(yaw) + y_scan * math.cos(yaw))
            if np.isnan(x) or np.isnan(y):
                continue
            # self.grid.update(x, y, occupied=True)
            self.grid.update_ray(x_robot, y_robot, x, y, last_cell_occupied)

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
    def __init__(self, size, resolution, origin, l_occ=0.85, l_free=-0.4, l_min=-5, l_max=5):
        self.size = size
        self.resolution = resolution
        self.width = int(size // resolution)
        self.height = int(size // resolution)
        self.origin = origin

        self.log_odds = np.zeros((self.height, self.width), dtype=np.float32)

        self.l_occ = l_occ
        self.l_free = l_free
        self.l_min = l_min
        self.l_max = l_max

        self.workspace_polygon = None
        self.workspace_mask = None

    def world_to_grid(self, x, y):
        x_index = int((x - self.origin[0]) // self.resolution)
        y_index = int((y - self.origin[1]) // self.resolution)
        return x_index, y_index

    def point_in_polygon(self, x, y, polygon):
        inside = False
        n = len(polygon)

        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]

            intersects = ((y1 > y) != (y2 > y))
            if intersects:
                x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if x < x_intersect:
                    inside = not inside

        return inside

    def set_workspace_polygon(self, polygon):
        self.workspace_polygon = polygon
        self.workspace_mask = np.zeros((self.height, self.width), dtype=bool)

        for y_idx in range(self.height):
            for x_idx in range(self.width):
                wx = self.origin[0] + (x_idx + 0.5) * self.resolution
                wy = self.origin[1] + (y_idx + 0.5) * self.resolution
                self.workspace_mask[y_idx, x_idx] = self.point_in_polygon(wx, wy, polygon)

    def initialize_outside_polygon_as_occupied(self):
        if self.workspace_mask is None:
            return
        self.log_odds[~self.workspace_mask] = self.l_max

    def is_inside_workspace_idx(self, x_idx, y_idx):
        if x_idx < 0 or x_idx >= self.width or y_idx < 0 or y_idx >= self.height:
            return False

        if self.workspace_mask is None:
            return True

        return self.workspace_mask[y_idx, x_idx]

    def update(self, x, y, occupied=True):
        if x < self.origin[0] or x > self.origin[0] + self.size:
            return
        if y < self.origin[1] or y > self.origin[1] + self.size:
            return

        x_index, y_index = self.world_to_grid(x, y)

        if not self.is_inside_workspace_idx(x_index, y_index):
            return

        if occupied:
            self.log_odds[y_index, x_index] += self.l_occ
        else:
            self.log_odds[y_index, x_index] += self.l_free

        self.log_odds[y_index, x_index] = np.clip(
            self.log_odds[y_index, x_index],
            self.l_min,
            self.l_max
        )

    def bresenham(self, x0, y0, x1, y1):
        cells = []

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1

        if dx > dy:
            err = dx / 2.0
            while x != x1:
                cells.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                cells.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy

        cells.append((x1, y1))
        return cells

    def update_ray(self, x_robot, y_robot, x_hit, y_hit, last_occupied=True):
        x0, y0 = self.world_to_grid(x_robot, y_robot)
        x1, y1 = self.world_to_grid(x_hit, y_hit)

        cells = self.bresenham(x0, y0, x1, y1)
        valid_cells = [(x, y) for (x, y) in cells if self.is_inside_workspace_idx(x, y)]

        if not valid_cells:
            return

        for i, (x, y) in enumerate(valid_cells):
            if last_occupied:
                if i == len(valid_cells) - 1:
                    self.log_odds[y, x] += self.l_occ
                else:
                    self.log_odds[y, x] += self.l_free
            else:
                self.log_odds[y, x] += self.l_free

            self.log_odds[y, x] = np.clip(
                self.log_odds[y, x],
                self.l_min,
                self.l_max
            )

    def get_data(self):
        probs = 1 - 1 / (1 + np.exp(self.log_odds))
        occupancy = (probs * 100).astype(np.int8)
        return occupancy.reshape(-1).tolist()