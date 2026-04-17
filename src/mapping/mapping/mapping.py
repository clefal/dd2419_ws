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
        # LaserScan input topic. Default assumes an upstream scan preprocessor node.
        self.declare_parameter("lidar_topic", "/localization/preprocessed_scan")
        # If true, the incoming LaserScan is assumed to already be filtered upstream.
        # If false, this node will filter ranges using `median_filter_scan()` before mapping.
        self.declare_parameter("input_is_preprocessed", True)
        self.declare_parameter("scans_to_skip", 3)
        self.declare_parameter("is_turning_topic", "/nav/is_turning")
        self.declare_parameter("workspace_topic", "/workspace")
        self.declare_parameter("grid_size", 12)
        self.declare_parameter("grid_origin", [-1.0, -1.0])


        # ------------------- CONFIG PARAMS ---------------
        # Grid params
        self.grid_size = self.get_parameter("grid_size").value
        self.grid_origin = self.get_parameter("grid_origin").value
        self.grid_resolution = self.get_parameter("grid_resolution").value
        self.worspace_topic = self.get_parameter("workspace_topic").value

        # Lidar params
        self.scans_to_skip = self.get_parameter("scans_to_skip").value
        self.input_is_preprocessed = bool(self.get_parameter("input_is_preprocessed").value)
        self.range_min = 0.1
        self.range_max = 5.0
        self.range_max_free_update = 3.0

        # Filter params
        self.median_filter_kernel_size = 5

        # Log-odds params
        self.log_odds_increse_occ = 0.9
        self.log_odds_decrease_free = -0.45
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

        # Workspace subscriber (get last published polygon)
        qos_workspace = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        workspace_topic = self.get_parameter("workspace_topic").value
        self.workspace_subscription = self.create_subscription(
            PolygonStamped,
            workspace_topic,
            self.workspace_callback,
            qos_workspace)
        self.workspace_subscription  # prevent unused variable warning

        # Is turning subscriber
        is_turning_topic = self.get_parameter("is_turning_topic").value
        self.is_turning_subscription = self.create_subscription(
            Bool,
            is_turning_topic,
            self.is_turning_callback,
            10)
        self.is_turning_subscription  # prevent unused variable warning
        self.is_turning = False

        # Grid (initialized once we receive the workspace polygon)
        self.grid = None
        self.workspace_received = False
        self.workspace_polygon_xy = None
        

        self.get_logger().info(
            f"Mapping node started | lidar_topic={lidar_topic}, "
            f"input_is_preprocessed={self.input_is_preprocessed}"
        )

        # TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

    def workspace_callback(self, msg: PolygonStamped):
        if not msg.polygon.points:
            self.get_logger().warn("Received workspace polygon with no points")
            return

        self.workspace_polygon_xy = [(float(p.x), float(p.y)) for p in msg.polygon.points]
        xs = [p[0] for p in self.workspace_polygon_xy]
        ys = [p[1] for p in self.workspace_polygon_xy]
        min_x, max_x = float(min(xs)), float(max(xs))
        min_y, max_y = float(min(ys)), float(max(ys))

        res = float(self.grid_resolution)
        if res <= 0.0:
            self.get_logger().error(f"Invalid grid_resolution={res}")
            return

        # Align bounds to resolution and ensure the max boundary is included.
        origin_x = math.floor(min_x / res) * res
        origin_y = math.floor(min_y / res) * res
        max_x_bound = math.ceil(max_x / res) * res
        max_y_bound = math.ceil(max_y / res) * res

        width = int(round((max_x_bound - origin_x) / res)) + 1
        height = int(round((max_y_bound - origin_y) / res)) + 1
        if width <= 0 or height <= 0:
            self.get_logger().error(f"Computed invalid grid size: width={width}, height={height}")
            return

        needs_reinit = (
            (self.grid is None)
            or (self.grid.width != width)
            or (self.grid.height != height)
            or (abs(self.grid.origin[0] - origin_x) > 1e-9)
            or (abs(self.grid.origin[1] - origin_y) > 1e-9)
        )
        if not needs_reinit:
            return

        if not self.workspace_received:
            self.get_logger().info(
                "Initializing occupancy grid from workspace polygon "
                f"(width={width}, height={height}, res={res}, origin=({origin_x:.3f},{origin_y:.3f}))"
            )
        else:
            self.get_logger().warn(
                "Workspace polygon changed; reinitializing occupancy grid "
                f"(width={width}, height={height}, res={res}, origin=({origin_x:.3f},{origin_y:.3f}))"
            )

        self.grid = OcupancyGridData(
            width=width,
            height=height,
            resolution=res,
            origin=[origin_x, origin_y],
            l_occ=self.log_odds_increse_occ,
            l_free=self.log_odds_decrease_free,
            l_min=self.log_odds_min,
            l_max=self.log_odds_max,
        )
        self.grid.set_workspace_polygon(self.workspace_polygon_xy)
        self.workspace_received = True
        self.publish_grid(self.get_clock().now().to_msg())

    def lidar_callback(self, msg: LaserScan):
        if self.is_turning:
            return
        if self.grid is None:
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
        except TransformException as ex:
            self.get_logger().warn(f"Async TF lookup failed: {ex}")
            return

        # Spin until transform found or `timeout_sec` seconds has passed
        rclpy.spin_until_future_complete(self, tf_future, timeout_sec=1)

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

        #self.get_logger().info(f"X: {x_robot}, Y: {y_robot}")

        filtered_ranges = (
            np.array(msg.ranges, dtype=float)
            if self.input_is_preprocessed
            else self.median_filter_scan(msg.ranges, kernel_size=self.median_filter_kernel_size)
        )
        last_cell_occupied = True

        for i in range(len(filtered_ranges)):
            r = filtered_ranges[i]
            last_cell_occupied = True

            if math.isinf(r) or math.isnan(r) or r > self.range_max :
                r = self.range_max_free_update
                last_cell_occupied = False
            
            if r < self.range_min:
                continue

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

        self.publish_grid(msg.header.stamp)

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

    def publish_grid(self, stamp):
        if self.grid is None:
            return

        occupancy_grid_msg = OccupancyGrid()
        occupancy_grid_msg.header.stamp = stamp
        occupancy_grid_msg.header.frame_id = "map"
        occupancy_grid_msg.info.resolution = self.grid.resolution
        occupancy_grid_msg.info.width = self.grid.width
        occupancy_grid_msg.info.height = self.grid.height
        occupancy_grid_msg.info.origin.position.x = self.grid.origin[0]
        occupancy_grid_msg.info.origin.position.y = self.grid.origin[1]
        occupancy_grid_msg.info.origin.position.z = 0.0
        occupancy_grid_msg.data = self.grid.get_data()

        self.ocuppancy_grid_publisher.publish(occupancy_grid_msg)

def main():
    rclpy.init()
    node = Mapping()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


class OcupancyGridData:
    def __init__(self, width, height, resolution, origin, l_occ=0.85, l_free=-0.4, l_min=-5, l_max=5):
        self.resolution = resolution
        self.width = int(width)
        self.height = int(height)
        self.origin = origin
        self.size_x = float(self.width) * float(self.resolution)
        self.size_y = float(self.height) * float(self.resolution)

        # Log-odds grid (float)
        self.log_odds = np.zeros((self.height, self.width), dtype=np.float32)
        self.inside_workspace_mask = None

        # Parameters
        self.l_occ = l_occ    # log odds increase for occupied
        self.l_free = l_free   # log odds decrease for free
        self.l_min = l_min
        self.l_max = l_max

    def set_workspace_polygon(self, pts_xy):
        """
        pts_xy: list of (x,y) points in world/map frame.
        Creates a boolean mask of which grid cells are inside the polygon.
        All cells outside are forced to occupied (log odds = l_max).
        """
        if not pts_xy or len(pts_xy) < 3:
            self.inside_workspace_mask = None
            return

        poly = [(float(x), float(y)) for (x, y) in pts_xy]

        def _point_on_segment(px, py, ax, ay, bx, by, eps=1e-9):
            abx = bx - ax
            aby = by - ay
            apx = px - ax
            apy = py - ay
            cross = abx * apy - aby * apx
            if abs(cross) > eps:
                return False
            dot = apx * abx + apy * aby
            if dot < -eps:
                return False
            sq_len = abx * abx + aby * aby
            if dot - sq_len > eps:
                return False
            return True

        def _point_in_poly(px, py, polygon):
            inside = False
            n = len(polygon)
            for i in range(n):
                x1, y1 = polygon[i]
                x2, y2 = polygon[(i + 1) % n]

                if _point_on_segment(px, py, x1, y1, x2, y2):
                    return True

                intersects = ((y1 > py) != (y2 > py)) and (
                    px < (x2 - x1) * (py - y1) / ((y2 - y1) if (y2 - y1) != 0.0 else 1e-12) + x1
                )
                if intersects:
                    inside = not inside
            return inside

        mask = np.zeros((self.height, self.width), dtype=bool)
        for gy in range(self.height):
            cy = self.origin[1] + (gy + 0.5) * self.resolution
            for gx in range(self.width):
                cx = self.origin[0] + (gx + 0.5) * self.resolution
                mask[gy, gx] = _point_in_poly(cx, cy, poly)

        self.inside_workspace_mask = mask
        self.log_odds[~self.inside_workspace_mask] = self.l_max

    def world_to_grid(self, x, y):
        x_index = int((x - self.origin[0]) // self.resolution)
        y_index = int((y - self.origin[1]) // self.resolution)
        return x_index, y_index

    def update(self, x, y, occupied=True):
        if x < self.origin[0] or x >= self.origin[0] + self.size_x:
            return
        if y < self.origin[1] or y >= self.origin[1] + self.size_y:
            return

        x_index, y_index = self.world_to_grid(x, y)

        if x_index < 0 or x_index >= self.width:
            return
        if y_index < 0 or y_index >= self.height:
            return
        if self.inside_workspace_mask is not None and not self.inside_workspace_mask[y_index, x_index]:
            return

        if occupied:
            self.log_odds[y_index, x_index] += self.l_occ
        else:
            self.log_odds[y_index, x_index] += self.l_free

        # Clamp
        self.log_odds[y_index, x_index] = np.clip(
            self.log_odds[y_index, x_index],
            self.l_min,
            self.l_max
        )

    def get_data(self):
        probs = 1 - 1 / (1 + np.exp(self.log_odds))  # sigmoid

        occupancy = (probs * 100).astype(np.int8)
        if self.inside_workspace_mask is not None:
            occupancy[~self.inside_workspace_mask] = 100

        return occupancy.reshape(-1).tolist()
    
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

        for i, (x, y) in enumerate(cells):
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                continue
            if self.inside_workspace_mask is not None and not self.inside_workspace_mask[y, x]:
                continue

            if last_occupied:
                # First cell → occupied
                if i == len(cells) - 1:
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
