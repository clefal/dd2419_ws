#!/usr/bin/env python3

import math
from typing import Tuple
import rclpy
from rclpy.node import Node

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

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


        path_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(Path, '/nav/global_path', self.path_callback, path_qos)

    
        # TF: map -> base_link
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._fixed_frame = 'map'
        self._base_frame = 'base_link'

        # Latest path (map frame)
        self._path_xy = []
        self._goal_yaw = None

        # Parameters
        self.declare_parameter('lookahead_distance', 0.3)        # m
        self.declare_parameter('nominal_linear_speed', 0.18)     # duty-equivalent
        self.declare_parameter('max_angular_speed', 0.22)        # duty-equivalent
        self.declare_parameter('goal_tolerance', 0.10)           # m
        self.declare_parameter('align_final_yaw', True)
        self.declare_parameter('steering_gain', 0.55)

        # Motor deadzone requirement: each wheel is 0 or |duty| >= this
        self._dc_min = 0.08

        # When to turn in place to reacquire path direction
        self._turn_in_place_yaw_thresh = 0.60  # rad
        self._yaw_tol = 0.05  # rad for final alignment

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

        # True when turning on the spot (opposite directions)
        turning_msg = Bool()
        turning_msg.data = (left * right < 0.0)
        self._turn_pub.publish(turning_msg)

    def stop(self):
        self.send_duty(0.0, 0.0)

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
        frame = (msg.header.frame_id or '').strip()
        self.get_logger().info(f"/nav/global_path received: frame='{frame}', poses={len(msg.poses)}")
        if frame != self._fixed_frame:
            self.get_logger().warn(
                f'Received /nav/global_path in frame "{frame}", expected "{self._fixed_frame}". Ignoring.'
            )
            self._path_xy = []
            self._goal_yaw = None
            self.publish_status('FAILED')
            return

        if len(msg.poses) == 0:
            self._path_xy = []
            self._goal_yaw = None
            self.publish_status('IDLE')
            self.get_logger().warn('Received empty /nav/global_path. Controller stopping until non-empty path arrives.')
            return

        self._path_xy = [(ps.pose.position.x, ps.pose.position.y) for ps in msg.poses]

        # Final yaw (planner now provides orientation)
        q = msg.poses[-1].pose.orientation
        self._goal_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]

        self.publish_status('RUNNING')

        sx, sy = self._path_xy[0]
        gx, gy = self._path_xy[-1]
        self.get_logger().info(
            f"Received path: {len(msg.poses)} poses, start=({sx:.2f},{sy:.2f}), goal=({gx:.2f},{gy:.2f}), goal_yaw={self._goal_yaw:.2f} rad"
        )

    # ----------------------------

    def _closest_path_index(self, rx: float, ry: float) -> int:
        best_i = 0
        best_d2 = float('inf')
        for i, (px, py) in enumerate(self._path_xy):
            dx = px - rx
            dy = py - ry
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        return best_i

    def _lookahead_point(self, rx: float, ry: float, lookahead: float):
        if not self._path_xy:
            return None

        i0 = self._closest_path_index(rx, ry)

        for i in range(i0, len(self._path_xy)):
            px, py = self._path_xy[i]
            if math.hypot(px - rx, py - ry) >= lookahead:
                return (px, py, i)

        px, py = self._path_xy[-1]
        return (px, py, len(self._path_xy) - 1)

    @staticmethod
    def enforce_motor_deadzone_pair(left: float, right: float, min_dc: float) -> Tuple[float, float]:
        """
        Requirement: each wheel is either 0 or |duty| >= min_dc.

        - If both wheels same direction and one is just under min_dc: scale BOTH up to preserve ratio.
        - If wheels opposite direction (turn-in-place): force each nonzero wheel to at least min_dc.
        - Finally: clamp tiny magnitudes to 0.
        """
        # Same direction (forward/back): scale both to keep ratio
        if left * right > 0.0:
            aL, aR = abs(left), abs(right)
            m = min(aL, aR)
            if 0.0 < m < min_dc:
                scale = min_dc / m
                left *= scale
                right *= scale

        # Opposite direction (turn in place): enforce minimum magnitude per wheel if nonzero
        if left * right < 0.0:
            if abs(left) > 0.0 and abs(left) < min_dc:
                left = math.copysign(min_dc, left)
            if abs(right) > 0.0 and abs(right) < min_dc:
                right = math.copysign(min_dc, right)

        # Per-wheel deadzone: tiny magnitudes become 0
        if 0.0 < abs(left) < min_dc:
            left = 0.0
        if 0.0 < abs(right) < min_dc:
            right = 0.0

        return left, right
    # ----------------------------

    def control_tick(self):
        # Empty path -> stop
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

        # Goal check
        gx, gy = self._path_xy[-1]
        goal_tol = float(self.get_parameter('goal_tolerance').value)
        dist_to_goal = math.hypot(gx - rx, gy - ry)

        if dist_to_goal <= goal_tol:
            if bool(self.get_parameter('align_final_yaw').value) and (self._goal_yaw is not None):
                yaw_err = wrap_angle(self._goal_yaw - ryaw)
                if abs(yaw_err) <= self._yaw_tol:
                    self.stop()
                    self.publish_status('REACHED')
                    self._path_xy = []
                    return

                wmax = float(self.get_parameter('max_angular_speed').value)
                # Simple proportional-in-duty turning in place
                w = clamp(yaw_err, -1.0, 1.0) * wmax
                left, right = self.enforce_motor_deadzone_pair(-w, w, self._dc_min)
                self.send_duty(left, right)
                return

            self.stop()
            self.publish_status('REACHED')
            self._path_xy = []
            return

        # Lookahead target
        lookahead = float(self.get_parameter('lookahead_distance').value)
        lookahead = max(0.05, lookahead)
        tgt = self._lookahead_point(rx, ry, lookahead)
        if tgt is None:
            self.stop()
            return

        tx, ty, _ = tgt

        # Target in robot frame
        dx = tx - rx
        dy = ty - ry
        cos_y = math.cos(ryaw)
        sin_y = math.sin(ryaw)
        x_r = cos_y * dx + sin_y * dy
        y_r = -sin_y * dx + cos_y * dy

        # If target behind / too misaligned -> turn in place
        heading_to_tgt = math.atan2(dy, dx)
        yaw_err = wrap_angle(heading_to_tgt - ryaw)
        if abs(yaw_err) > self._turn_in_place_yaw_thresh or x_r < 0.05:
            wmax = float(self.get_parameter('max_angular_speed').value)
            w = clamp(yaw_err, -1.0, 1.0) * wmax
            left, right = self.enforce_motor_deadzone_pair(-w, w, self._dc_min)
            self.send_duty(left, right)
            return

        # Pure Pursuit curvature: kappa = 2*y_r / L^2
        kappa = (2.0 * y_r) / (lookahead * lookahead)

        v_nom = float(self.get_parameter('nominal_linear_speed').value)
        wmax = float(self.get_parameter('max_angular_speed').value)
        k_steer = float(self.get_parameter('steering_gain').value)

        # Slow down in curves (simple, stable indoors)
        v = v_nom / (1.0 + 3.0 * abs(kappa))
        v = clamp(v, 0.0, v_nom)

        # Steering
        w = k_steer * v * kappa
        w = clamp(w, -wmax, wmax)

        # Convert to wheel duties
        left = v - w
        right = v + w

        # Enforce only the actuator requirement at the wheel level
        left, right = self.enforce_motor_deadzone_pair(left, right, self._dc_min)

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
