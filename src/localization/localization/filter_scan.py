import rclpy
import math
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np


def neighbor_distance(r1: float, r2: float, cos_dtheta: float) -> float:
    d2 = r1 * r1 + r2 * r2 - 2.0 * r1 * r2 * cos_dtheta
    return math.sqrt(max(d2, 0.0))


class ScanPreprocessor(Node):
    def __init__(self):
        super().__init__('scan_preprocessor')

        self.declare_parameter('input_topic', '/lidar/scan')
        self.declare_parameter('output_topic', '/localization/preprocessed_scan')
        

        # Preprocessing
        # Maximum range kept when turning scan beams into points for mapping and ICP.
        self.declare_parameter("range_max_clip", 7.0)
        # Maximum range kept in the published preprocessed scan message.
        self.declare_parameter("range_max_filter_scan", 7.0)
        # Median filter size applied to the raw range array.
        self.declare_parameter("median_kernel_size", 3)
        # Reject a beam if it differs from both adjacent beams by more than this range jump.
        self.declare_parameter("range_jump_thresh", 0.20)
        # Remove points whose immediate scan-order neighbors are both farther than this distance.
        self.declare_parameter("neighbor_dist_thresh", 0.08)

        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.range_max_clip = float(self.get_parameter("range_max_clip").value)
        self.range_max_filter_scan = float(self.get_parameter("range_max_filter_scan").value)
        self.median_kernel_size = int(self.get_parameter("median_kernel_size").value)
        self.range_jump_thresh = float(self.get_parameter("range_jump_thresh").value)
        self.neighbor_dist_thresh = float(self.get_parameter("neighbor_dist_thresh").value)

        self.pub = self.create_publisher(LaserScan, self.output_topic, 10)
        self.sub = self.create_subscription(LaserScan, self.input_topic, self.callback, 10)

        self.get_logger().info('Scan preprocessor started and listening to ' + self.input_topic + ' on ' + self.output_topic)

    def callback(self, msg):
        self.preprocess_scan(msg)

    def preprocess_scan(self, scan: LaserScan) -> LaserScan:
        filtered_scan = LaserScan()
        filtered_scan.header = scan.header
        filtered_scan.angle_min = scan.angle_min
        filtered_scan.angle_max = scan.angle_max
        filtered_scan.angle_increment = scan.angle_increment
        filtered_scan.time_increment = scan.time_increment
        filtered_scan.scan_time = scan.scan_time
        filtered_scan.range_min = scan.range_min
        filtered_scan.range_max = min(scan.range_max, self.range_max_filter_scan)
        filtered_scan.intensities = scan.intensities

        filtered_ranges = self.median_filter_ranges(scan.ranges, kernel_size=self.median_kernel_size)
        filtered_ranges = self.reject_range_spikes(filtered_ranges, jump_thresh=self.range_jump_thresh)

        filtered_ranges = self.reject_isolated_points(
            filtered_ranges,
            scan,
            neighbor_dist_thresh=self.neighbor_dist_thresh,
        )
        
        filtered_ranges[
            np.logical_and(np.isfinite(filtered_ranges), filtered_ranges > self.range_max_filter_scan)
        ] = np.inf

        filtered_scan.ranges = filtered_ranges.tolist()
        self.pub.publish(filtered_scan)
        return filtered_scan
    
    def median_filter_ranges(self, ranges, kernel_size: int = 5) -> np.ndarray:
        kernel_size = max(1, int(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1

        half = kernel_size // 2
        arr = np.array(ranges, dtype=float)
        out = arr.copy()

        for i in range(arr.shape[0]):
            vals = []
            for j in range(max(0, i - half), min(arr.shape[0], i + half + 1)):
                v = arr[j]
                if math.isfinite(v):
                    vals.append(v)
            out[i] = float(np.median(vals)) if vals else np.nan

        return out

    def reject_isolated_points(
        self,
        ranges,
        scan: LaserScan,
        neighbor_dist_thresh: float = 0.12,
    ) -> np.ndarray:
        out = np.array(ranges, dtype=float)
        if out.shape[0] < 3 or neighbor_dist_thresh <= 0.0:
            return out

        max_valid_range = min(scan.range_max, self.range_max_clip, self.range_max_filter_scan)
        valid = np.logical_and.reduce((
            np.isfinite(out),
            out >= max(scan.range_min, 0.05),
            out <= max_valid_range,
        ))

        keep = np.zeros(out.shape[0], dtype=bool)
        cos_dtheta = math.cos(scan.angle_increment)

        if valid[0] and valid[1]:
            keep[0] = neighbor_distance(out[0], out[1], cos_dtheta) < neighbor_dist_thresh

        if valid[-1] and valid[-2]:
            keep[-1] = neighbor_distance(out[-1], out[-2], cos_dtheta) < neighbor_dist_thresh

        for i in range(1, out.shape[0] - 1):
            if not valid[i]:
                continue

            close_prev = valid[i - 1] and (
                neighbor_distance(out[i], out[i - 1], cos_dtheta) < neighbor_dist_thresh
            )
            close_next = valid[i + 1] and (
                neighbor_distance(out[i], out[i + 1], cos_dtheta) < neighbor_dist_thresh
            )
            keep[i] = close_prev or close_next

        out[np.logical_and(valid, ~keep)] = np.nan
        return out


    def reject_range_spikes(self, ranges, jump_thresh: float = 0.25) -> np.ndarray:
        arr = np.array(ranges, dtype=float)
        out = arr.copy()

        if arr.shape[0] < 3 or jump_thresh <= 0.0:
            return out

        for i in range(1, arr.shape[0] - 1):
            cur = arr[i]
            prev = arr[i - 1]
            nxt = arr[i + 1]

            if not math.isfinite(cur):
                continue

            if math.isfinite(prev) and math.isfinite(nxt):
                if abs(cur - prev) > jump_thresh and abs(cur - nxt) > jump_thresh:
                    out[i] = np.nan

        return out



def main():
    rclpy.init()
    node = ScanPreprocessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()
