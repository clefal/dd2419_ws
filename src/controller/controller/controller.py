#!/usr/bin/env python3

import math
from typing import Tuple, Optional, List

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from std_msgs.msg import String, Bool, Float32
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
        self.create_subscription(Float32, '/nav/backup_distance', self.backup_callback, 10)

    
        # TF: map -> base_link
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._fixed_frame = 'map'
        self._base_frame = 'base_link'

        # Latest path (map frame)
        self._path_xy = []
        self._goal_yaw = None

        # Parameters
        self.declare_parameter('lookahead_distance', 0.4)        # m
        self.declare_parameter('nominal_linear_speed', 0.35)     # duty-equivalent
        self.declare_parameter('max_angular_speed', 0.2)        # duty-equivalent
        self.declare_parameter('goal_tolerance', 0.05)           # m
        self.declare_parameter('align_final_yaw', True)
        self.declare_parameter('steering_gain', 0.4)


        self.declare_parameter('goal_slow_radius', 0.40)         # m (start slowing within this distance)
        self.declare_parameter('min_linear_speed', 0.12)         # duty-equivalent (keep > deadzone margin)
        self.declare_parameter('turn_gain', 0.5)                 # duty-per-rad for in-place turning
        self.declare_parameter('control_period', 0.02)           # s (0.05=20Hz, 0.1=10Hz)
        
        self.declare_parameter('vel_filter_alpha', 0.65)         # 0..1 (higher = more smoothing)
        self.declare_parameter('yaw_filter_alpha', 0.70)         # 0..1
        self.declare_parameter('min_turn_duty', 0.10)            # duty-equivalent for in-place turning (>= deadzone margin)


        # Motor deadzone requirement: each wheel is 0 or |duty| >= this
        self._dc_min = 0.08

        # When to turn in place to reacquire path direction
        self._turn_in_place_yaw_thresh = 0.60  # rad
        self._yaw_tol = 0.05  # rad for final alignment

        # State for filtering
        self._v_filt = 0.0
        self._w_filt = 0.0

        # Path progress (avoid snapping backwards)
        self._last_path_idx = 0
        self._closest_search_window = 40  # points forward to search

        # Control loop
              
        period = float(self.get_parameter('control_period').value)
        self._timer = self.create_timer(period, self.control_tick)
        
        self._backup_active = False
        self._backup_target_m = 0.0
        self._backup_start_xy = None
        self._backup_duty = 0.12

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

        q = msg.poses[-1].pose.orientation
        self._goal_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]

        self._last_path_idx = 0
        self._v_filt = 0.0
        self._w_filt = 0.0

        self.publish_status('RUNNING')

        sx, sy = self._path_xy[0]
        gx, gy = self._path_xy[-1]
        self.get_logger().info(
            f"Received path: {len(msg.poses)} poses, start=({sx:.2f},{sy:.2f}), goal=({gx:.2f},{gy:.2f}), goal_yaw={self._goal_yaw:.2f} rad"
        )

    def backup_callback(self, msg: Float32):
        d = float(msg.data)
        if d <= 0.0:
            self.get_logger().warn(f'Ignoring non-positive backup distance: {d:.3f}')
            return

        self._backup_active = True
        self._backup_target_m = d
        self._backup_start_xy = None
        self._path_xy = []
        self._goal_yaw = None
        self.publish_status('RUNNING')
        self.get_logger().info(f'Starting backup maneuver: {d:.3f} m')

    # ----------------------------

    def _closest_path_index(self, rx: float, ry: float) -> int:
        """Closest index search constrained to a forward window to prevent jumping backwards."""
        if not self._path_xy:
            return 0

        lo = self._last_path_idx
        hi = min(len(self._path_xy) - 1, lo + self._closest_search_window)

        best_i = lo
        best_d2 = float('inf')
        for i in range(lo, hi + 1):
            px, py = self._path_xy[i]
            dx = px - rx
            dy = py - ry
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_i = i

        self._last_path_idx = best_i
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

    # ----------------------------

    def _lowpass(self, prev: float, new: float, alpha: float) -> float:
        alpha = clamp(alpha, 0.0, 0.98)
        return alpha * prev + (1.0 - alpha) * new

    def _apply_deadzone_smooth(self, left: float, right: float, v_sign: float) -> Tuple[float, float]:
        """
        Smooth-ish deadzone handling:
        - For normal driving (same direction): if one wheel would drop below dc_min, add a small equal "boost"
          to both wheels to lift the smaller one over the threshold while preserving (right-left) turning difference.
        - For turning in place (opposite directions): enforce minimum magnitude per wheel (needed to actually rotate).
        - Clamp tiny residuals to 0.
        """
        min_dc = self._dc_min

        # Turn in place: ensure both wheels exceed deadzone
        if left * right < 0.0:
            if abs(left) > 0.0:
                left = math.copysign(max(abs(left), min_dc), left)
            if abs(right) > 0.0:
                right = math.copysign(max(abs(right), min_dc), right)
            return clamp(left, -1.0, 1.0), clamp(right, -1.0, 1.0)

        # Same direction: lift the smaller wheel with a shared additive boost (preserves curvature)
        if left * right > 0.0:
            aL, aR = abs(left), abs(right)
            m = min(aL, aR)
            if 0.0 < m < min_dc:
                boost = (min_dc - m)
                left += v_sign * boost
                right += v_sign * boost

        # Anything still under deadzone -> 0 (avoid buzzing)
        if 0.0 < abs(left) < min_dc:
            left = 0.0
        if 0.0 < abs(right) < min_dc:
            right = 0.0

        return clamp(left, -1.0, 1.0), clamp(right, -1.0, 1.0)

    # ----------------------------

    def control_tick(self):
        # Backup maneuver (simple reverse)
        if self._backup_active:
            pose = self.get_pose_2d()
            if pose is None:
                self.stop()
                self.publish_status('FAILED')
                self.get_logger().warn('Backup failed: no TF pose available (map->base_link).')
                self._backup_active = False
                return

            rx, ry, _ = pose
            if self._backup_start_xy is None:
                self._backup_start_xy = (rx, ry)

            sx, sy = self._backup_start_xy
            moved = math.hypot(rx - sx, ry - sy)
            if moved >= self._backup_target_m:
                self.stop()
                self._backup_active = False
                self.publish_status('REACHED')
                self.get_logger().info(
                    f'Backup complete: target={self._backup_target_m:.3f} m, moved={moved:.3f} m'
                )
                return

            left = -self._backup_duty
            right = -self._backup_duty
            v_sign = -1.0
            left, right = self._apply_deadzone_smooth(left, right, v_sign)
            self.send_duty(left, right)
            return

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

        # Final alignment (if within goal radius)
        if dist_to_goal <= goal_tol:
            if bool(self.get_parameter('align_final_yaw').value) and (self._goal_yaw is not None):
                yaw_err = wrap_angle(self._goal_yaw - ryaw)
                if abs(yaw_err) <= self._yaw_tol:
                    self.stop()
                    self.publish_status('REACHED')
                    self._path_xy = []
                    return

                wmax = float(self.get_parameter('max_angular_speed').value)
                k_turn = float(self.get_parameter('turn_gain').value)
                min_turn = float(self.get_parameter('min_turn_duty').value)
                min_turn = max(self._dc_min + 0.02, min_turn)

                w = clamp(k_turn * yaw_err, -wmax, wmax)
                # ensure it actually turns (overcomes stiction) but still proportional near zero
                if abs(w) > 0.0:
                    w = math.copysign(max(abs(w), min_turn), w)

                left = -w
                right = w
                left, right = self._apply_deadzone_smooth(left, right, v_sign=0.0)
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

        # If target behind / too misaligned -> turn in place to reacquire
        heading_to_tgt = math.atan2(dy, dx)
        yaw_err = wrap_angle(heading_to_tgt - ryaw)
        if abs(yaw_err) > self._turn_in_place_yaw_thresh or x_r < 0.05:
            wmax = float(self.get_parameter('max_angular_speed').value)
            k_turn = float(self.get_parameter('turn_gain').value)
            min_turn = float(self.get_parameter('min_turn_duty').value)
            min_turn = max(self._dc_min + 0.02, min_turn)

            w = clamp(k_turn * yaw_err, -wmax, wmax)
            if abs(w) > 0.0:
                w = math.copysign(max(abs(w), min_turn), w)

            left = -w
            right = w
            left, right = self._apply_deadzone_smooth(left, right, v_sign=0.0)
            self.send_duty(left, right)
            return

        # Pure Pursuit curvature: kappa = 2*y_r / L^2
        kappa = (2.0 * y_r) / (lookahead * lookahead)

        v_nom = float(self.get_parameter('nominal_linear_speed').value)
        wmax = float(self.get_parameter('max_angular_speed').value)
        k_steer = float(self.get_parameter('steering_gain').value)

        # --- Speed shaping ---
        # 1) Curve-based slowdown
        v_curve = v_nom / (1.0 + 3.0 * abs(kappa))
        v_curve = clamp(v_curve, 0.0, v_nom)

        # 2) Goal-approach slowdown (smooth stop)
        slow_radius = float(self.get_parameter('goal_slow_radius').value)
        v_min = float(self.get_parameter('min_linear_speed').value)
        slow_radius = max(0.05, slow_radius)
        v_min = max(self._dc_min + 0.02, min(v_min, v_nom))

        approach = clamp(dist_to_goal / slow_radius, 0.0, 1.0)
        v_goal = v_min + (v_nom - v_min) * approach

        v_cmd = min(v_curve, v_goal)

        # --- Steering ---
        w_cmd = k_steer * v_cmd * kappa
        w_cmd = clamp(w_cmd, -wmax, wmax)

        # --- Filtering (optional but recommended) ---
        a_v = float(self.get_parameter('vel_filter_alpha').value)
        a_w = float(self.get_parameter('yaw_filter_alpha').value)
        self._v_filt = self._lowpass(self._v_filt, v_cmd, a_v)
        self._w_filt = self._lowpass(self._w_filt, w_cmd, a_w)

        v = self._v_filt
        w = self._w_filt

        # Convert to wheel duties
        left = v - w
        right = v + w

        # Deadzone smoothing / compensation
        v_sign = 1.0 if v >= 0.0 else -1.0
        left, right = self._apply_deadzone_smooth(left, right, v_sign=v_sign)

        self.send_duty(left, right)


def main():
    rclpy.init()
    node = Controller()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.stop()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()