#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Bool
from nav_msgs.msg import Path
from robp_interfaces.msg import DutyCycles

from tf2_ros import Buffer, TransformListener
from tf_transformations import euler_from_quaternion


def wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class Controller(Node):

    def __init__(self):
        super().__init__('controller')

        # Publishers / subscribers
        self._cmd_pub = self.create_publisher(DutyCycles, '/phidgets/motor/duty_cycles', 10)
        self._status_pub = self.create_publisher(String, '/nav/status', 10)
        self._turn_pub = self.create_publisher(Bool, '/nav/is_turning', 10)

        self.create_subscription(Path, '/nav/global_path', self.path_callback, 10)

        # TF to get robot pose (map -> base_link)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._fixed_frame = 'map'
        self._base_frame = 'base_link'

        # Latest path (stored as list of (x, y) in fixed frame)
        self._path_xy = []
        self._path_frame = None
        self._last_path_stamp = None

        # ----------------------------
        # Parameters 

        self.declare_parameter('lookahead_distance', 0.7)      # m
        self.declare_parameter('nominal_linear_speed', 0.18)    # duty-equivalent
        self.declare_parameter('max_angular_speed', 0.22)       # duty-equivalent
        self.declare_parameter('goal_tolerance', 0.10)          # m
        self.declare_parameter('align_final_yaw', True)    
        self.declare_parameter('steering_gain', 0.55)

        # Minimal tuning (duty cycles) 
        self._v_min = 0.08   # motors might not actuate below this
        self._w_min = 0.09   # min turning-on-spot command
        self._yaw_tol = 0.05 # rad, used only if align_final_yaw=True

        # Turn-in-place behavior threshold
        self._turn_in_place_yaw_thresh = 0.60  # rad

        # Control loop
        self._timer = self.create_timer(0.1, self.control_tick)  # 10 Hz

    # ----------------------------

    def publish_status(self, s: str):
        msg = String()
        msg.data = s
        self._status_pub.publish(msg)

    def send_duty(self, left: float, right: float):
        m = DutyCycles()
        m.duty_cycle_left = float(clamp(left, -1.0, 1.0))
        m.duty_cycle_right = float(clamp(right, -1.0, 1.0))
        self._cmd_pub.publish(m)

        # Publish turning status (True if wheels opposite directions)
        turning_msg = Bool()
        turning_msg.data = (left * right < 0.0)
        self._turn_pub.publish(turning_msg)

    def stop(self):
        self.send_duty(0.0, 0.0)

    # ----------------------------

    def get_pose_2d(self):
        try:
            t = self._tf_buffer.lookup_transform(self._fixed_frame, self._base_frame, rclpy.time.Time())
        except Exception:
            return None

        x = t.transform.translation.x
        y = t.transform.translation.y
        q = t.transform.rotation
        yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        return x, y, yaw

    def path_callback(self, msg: Path):
        frame = msg.header.frame_id.strip() if msg.header.frame_id else ''
        if frame == '':
            self.get_logger().warn('Received /nav/global_path with empty frame_id; ignoring.')
            return

        self._path_frame = frame
        self._last_path_stamp = msg.header.stamp

        if len(msg.poses) == 0:
            self._path_xy = []
            self.publish_status('IDLE')
            return

        # If path is not in fixed_frame, try to transform points to fixed_frame using TF.
        # If TF not available, reject with warning (per requirements).
        if frame != self._fixed_frame:
            tf = self._lookup_tf_2d(self._fixed_frame, frame)
            if tf is None:
                self.get_logger().warn(
                    f'Path frame mismatch: path in "{frame}", controller fixed_frame "{self._fixed_frame}", '
                    f'and TF is unavailable. Ignoring path.'
                )
                self._path_xy = []
                self.publish_status('FAILED')
                return

            tx, ty, tyaw = tf
            out = []
            for ps in msg.poses:
                px = ps.pose.position.x
                py = ps.pose.position.y
                # rotate + translate from "frame" into fixed_frame
                x = tx + (px * math.cos(tyaw) - py * math.sin(tyaw))
                y = ty + (px * math.sin(tyaw) + py * math.cos(tyaw))
                out.append((x, y))
            self._path_xy = out
        else:
            self._path_xy = [(ps.pose.position.x, ps.pose.position.y) for ps in msg.poses]

        self.publish_status('RUNNING')

    def _lookup_tf_2d(self, target_frame: str, source_frame: str):
        """
        Returns (tx, ty, tyaw) for transform target_frame <- source_frame, or None.
        """
        try:
            t = self._tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
        except Exception:
            return None

        tx = t.transform.translation.x
        ty = t.transform.translation.y
        q = t.transform.rotation
        tyaw = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        return (tx, ty, tyaw)

    # ----------------------------

    def _closest_path_index(self, rx: float, ry: float) -> int:
        """
        Returns index of closest path point to robot.
        """
        best_i = 0
        best_d2 = float('inf')
        for i, (px, py) in enumerate(self._path_xy):
            dx = px - rx
            dy = py - ry
            d2 = dx*dx + dy*dy
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        return best_i

    def _lookahead_point(self, rx: float, ry: float, lookahead: float):
        """
        Pure Pursuit target selection:
        1) find closest point index on path
        2) walk forward until distance >= lookahead
        If no such point exists, return final path point.
        Returns (tx, ty, idx) or None if path empty.
        """
        if not self._path_xy:
            return None

        i0 = self._closest_path_index(rx, ry)

        # Walk forward to the first point at/after lookahead distance
        for i in range(i0, len(self._path_xy)):
            px, py = self._path_xy[i]
            if math.hypot(px - rx, py - ry) >= lookahead:
                return (px, py, i)

        # Otherwise use the last point
        px, py = self._path_xy[-1]
        return (px, py, len(self._path_xy) - 1)

    # ----------------------------

    def control_tick(self):
        # Edge case: no path -> stop motors
        if not self._path_xy:
            self.stop()
            return

        pose = self.get_pose_2d()
        if pose is None:
            self.stop()
            self.publish_status('FAILED')
            self.get_logger().warn('No TF pose available (map->base_link).')
            return

        rx, ry, ryaw = pose

        # Stop if close to end of path
        gx, gy = self._path_xy[-1]
        goal_tol = float(self.get_parameter('goal_tolerance').value)
        dist_to_goal = math.hypot(gx - rx, gy - ry)

        if dist_to_goal <= goal_tol:
            if bool(self.get_parameter('align_final_yaw').value):
                # Optional: align to final pose yaw if available (best-effort)
                # We only have (x,y) stored, so we attempt to align to heading of final segment.
                if len(self._path_xy) >= 2:
                    x2, y2 = self._path_xy[-1]
                    x1, y1 = self._path_xy[-2]
                    desired_yaw = math.atan2(y2 - y1, x2 - x1)
                    yaw_err = wrap_angle(desired_yaw - ryaw)
                    if abs(yaw_err) <= self._yaw_tol:
                        self.stop()
                        self.publish_status('REACHED')
                        self._path_xy = []
                        return
                    wmax = float(self.get_parameter('max_angular_speed').value)
                    w = clamp(yaw_err, -1.0, 1.0)  # normalized-ish before scaling below
                    w = clamp(w * wmax, -wmax, wmax)
                    if abs(w) < self._w_min:
                        w = math.copysign(self._w_min, w)
                    self.send_duty(-w, w)
                    return

            self.stop()
            self.publish_status('REACHED')
            self._path_xy = []
            return

        # Pure Pursuit target selection
        lookahead = float(self.get_parameter('lookahead_distance').value)
        lookahead = max(0.05, lookahead)  # avoid degenerate division
        tgt = self._lookahead_point(rx, ry, lookahead)
        if tgt is None:
            self.stop()
            return
        tx, ty, _ = tgt

        # Transform target point into robot frame (x_r forward, y_r left)
        dx = tx - rx
        dy = ty - ry
        cos_y = math.cos(ryaw)
        sin_y = math.sin(ryaw)
        x_r = cos_y * dx + sin_y * dy
        y_r = -sin_y * dx + cos_y * dy

        # If the target is "behind" us, rotate in place to reacquire path direction
        heading_to_tgt = math.atan2(dy, dx)
        yaw_err = wrap_angle(heading_to_tgt - ryaw)
        if abs(yaw_err) > self._turn_in_place_yaw_thresh or x_r < 0.05:
            wmax = float(self.get_parameter('max_angular_speed').value)
            w = clamp(yaw_err, -1.0, 1.0)
            w = clamp(w * wmax, -wmax, wmax)
            if abs(w) < self._w_min:
                w = math.copysign(self._w_min, w)
            self.send_duty(-w, w)
            return

        # Pure Pursuit curvature kappa = 2*y_r / L^2
        kappa = (2.0 * y_r) / (lookahead * lookahead)

        # Command: w = v * kappa
        v_nom = float(self.get_parameter('nominal_linear_speed').value)
        wmax = float(self.get_parameter('max_angular_speed').value)

        # Slightly reduce v when curvature is high
        v = v_nom / (1.0 + 3 * abs(kappa))
        v = clamp(v, 0.0, v_nom)

        k_steer = float(self.get_parameter('steering_gain').value)
        w = k_steer * v * kappa
        w = clamp(w, -wmax, wmax)

        # Enforce minimum effective commands (duty-cycle domain)
        if v > 0.0 and v < self._v_min:
            v = self._v_min
        if abs(w) > 0.0 and abs(w) < self._w_min:
            w = math.copysign(self._w_min, w)

        # Convert (v, w) to left/right duty cycles 
        left = v - w
        right = v + w
        self.send_duty(left, right)


def main():
    rclpy.init()
    node = Controller()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.stop()
        except Exception:
            pass
    rclpy.shutdown()


if __name__ == '__main__':
    main()