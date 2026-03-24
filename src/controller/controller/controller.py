#!/usr/bin/env python3

import json
import math
import time
from typing import Optional, Tuple
import rclpy
from rclpy.node import Node

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from std_msgs.msg import String, Bool, Float32, Int64
from nav_msgs.msg import Path
from robp_interfaces.msg import DutyCycles
from robp_interfaces.srv import GetPosOfObj

from tf2_ros import Buffer, TransformListener
from tf_transformations import euler_from_quaternion

from .final_approach_controller import FinalApproachController


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
        self.create_subscription(Bool, '/nav/final_approach/enable', self.final_approach_enable_callback, 10)
        self.create_subscription(Int64, '/nav/final_approach/target_id', self.final_approach_target_id_callback, 10)


    
        # TF: map -> base_link
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._fixed_frame = 'map'
        self._base_frame = 'base_link'

        # Latest path (map frame)
        self._path_xy = []
        self._goal_yaw = None

        self._final_approach_enabled = False
        self._final_target_id = None
        self._final_target_xy: Optional[Tuple[float, float]] = None
        self._final_target_last_seen_wall: Optional[float] = None
        self._final_target_request_pending = False
        self._final_target_request_period = 0.1
        self._final_target_request_last_wall = 0.0

        self._get_pos_of_obj_client = self.create_client(GetPosOfObj, 'object_manager/get_pos_of_obj')
        while not self._get_pos_of_obj_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('get_pos_of_obj service not available, waiting again...')

        # Parameters
        self.declare_parameter('lookahead_distance', 0.3)        # m
        self.declare_parameter('nominal_linear_speed', 0.5)     # 0.35 duty-equivalent
        self.declare_parameter('max_angular_speed', 0.3)        # 0.2 duty-equivalent
        self.declare_parameter('goal_tolerance', 0.08)  #0.05         # m
        self.declare_parameter('align_final_yaw', True)
        self.declare_parameter('steering_gain', 0.3)


        self.declare_parameter('goal_slow_radius', 0.40)         # m (start slowing within this distance)
        self.declare_parameter('min_linear_speed', 0.12)         # duty-equivalent (keep > deadzone margin)
        self.declare_parameter('turn_gain', 0.3)                 # duty-per-rad for in-place turning
        self.declare_parameter('control_period', 0.1)           # s (0.05=20Hz, 0.1=10Hz)
        self.declare_parameter('wheel_slew_rate', 1.5)          # duty/s max per-wheel change (except stop)

        self.declare_parameter('final_nominal_speed', 0.16)                # duty-equivalent for close approach
        self.declare_parameter('final_turn_gain', 0.8)                     # steering gain during close approach
        self.declare_parameter('final_max_angular_speed', 0.18)            # keep final approach conservative
        self.declare_parameter('final_turn_in_place_yaw_thresh', 0.35)     # rad
        self.declare_parameter('final_stop_distance', 0.17)                # m
        self.declare_parameter('final_target_timeout', 1.5)                # s


        # Motor deadzone requirement: each wheel is 0 or |duty| >= this
        self._dc_min = 0.08

        # When to turn in place to reacquire path direction
        self._turn_in_place_yaw_thresh = 0.75  # rad
        self._yaw_tol = 0.05  # rad for final alignment

        # Control loop
              
        period = float(self.get_parameter('control_period').value)
        self._control_period = period
        self._timer = self.create_timer(period, self.control_tick)
        self._last_left_cmd = 0.0
        self._last_right_cmd = 0.0
        
        self._backup_active = False
        self._backup_target_m = 0.0
        self._backup_start_xy = None
        self._backup_duty = 0.12

        self._final_controller = FinalApproachController(
            nominal_speed=float(self.get_parameter('final_nominal_speed').value),
            turn_gain=float(self.get_parameter('final_turn_gain').value),
            max_angular_speed=float(self.get_parameter('final_max_angular_speed').value),
            turn_in_place_yaw_thresh=float(self.get_parameter('final_turn_in_place_yaw_thresh').value),
            stop_distance=float(self.get_parameter('final_stop_distance').value),
            min_wheel_duty=self._dc_min,
        )

    # ----------------------------

    def publish_status(self, s: str):
        msg = String()
        msg.data = s
        self._status_pub.publish(msg)

    def send_duty(self, left: float, right: float):
        target_left = float(clamp(left, -1.0, 1.0))
        target_right = float(clamp(right, -1.0, 1.0))

        # Keep stops immediate for safety; otherwise limit per-tick duty jumps.
        if target_left == 0.0 and target_right == 0.0:
            out_left = 0.0
            out_right = 0.0
        else:
            slew_rate = float(self.get_parameter('wheel_slew_rate').value)
            max_delta = max(0.0, slew_rate) * self._control_period
            out_left = clamp(target_left, self._last_left_cmd - max_delta, self._last_left_cmd + max_delta)
            out_right = clamp(target_right, self._last_right_cmd - max_delta, self._last_right_cmd + max_delta)

        m = DutyCycles()
        m.duty_cycle_left = out_left
        m.duty_cycle_right = out_right
        self._cmd_pub.publish(m)
        self._last_left_cmd = out_left
        self._last_right_cmd = out_right

        # True when turning on the spot (opposite directions)
        turning_msg = Bool()
        turning_msg.data = (out_left * out_right < 0.0)
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

        self._final_approach_enabled = False
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
        self._final_approach_enabled = False
        self.publish_status('RUNNING')
        self.get_logger().info(f'Starting backup maneuver: {d:.3f} m')

    def final_approach_enable_callback(self, msg: Bool):
        self._final_approach_enabled = bool(msg.data)
        if self._final_approach_enabled:
            self._path_xy = []
            self._goal_yaw = None
            self._final_target_xy = None
            self._final_target_last_seen_wall = None
            self._final_target_request_pending = False
            self._final_target_request_last_wall = 0.0
            self.publish_status('RUNNING')
            self.get_logger().info('Final approach enabled.')
        else:
            self.stop()
            self._final_target_request_pending = False
            self.get_logger().info('Final approach disabled.')

    def final_approach_target_id_callback(self, msg: Int64):
        self._final_target_id = int(msg.data)
        self._final_target_xy = None
        self._final_target_last_seen_wall = None
        self._final_target_request_pending = False
        self._final_target_request_last_wall = 0.0
        self.get_logger().info(f'Final approach target id set to: {self._final_target_id}')

    def request_final_target_pose(self) -> None:
        if self._final_target_id is None or self._final_target_request_pending:
            return

        now = time.time()
        if (now - self._final_target_request_last_wall) < self._final_target_request_period:
            return

        req = GetPosOfObj.Request()
        req.obj_id = int(self._final_target_id)

        self._final_target_request_pending = True
        self._final_target_request_last_wall = now
        future = self._get_pos_of_obj_client.call_async(req)
        future.add_done_callback(self.final_target_pose_response_callback)

    def final_target_pose_response_callback(self, future) -> None:
        self._final_target_request_pending = False
        try:
            res = future.result()
        except Exception as exc:
            self.get_logger().warn(
                f'Failed to refresh pose for final target {self._final_target_id}: {exc}'
            )
            return

        self._final_target_xy = (res.obj_x, res.obj_y)
        self._final_target_last_seen_wall = time.time()



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

    def enforce_motor_deadzone_pair(self, left: float, right: float, min_dc: float) -> Tuple[float, float]:
        """
        Affine deadzone remap:
        each non-zero wheel command in [0..1] is remapped to [min_dc..1].
        This keeps command output continuous and avoids repeated near-threshold lifting.
        """
        eps = 1e-4

        def remap(dc: float) -> float:
            a = abs(dc)
            if a <= eps:
                return 0.0
            a = clamp(a, 0.0, 1.0)
            # 0% input -> min_dc, 100% input -> 1.0
            a = min_dc + (1.0 - min_dc) * a
            return math.copysign(a, dc)

        return remap(left), remap(right)
    # ----------------------------

    def control_tick(self):
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

            left, right = self.enforce_motor_deadzone_pair(-self._backup_duty, -self._backup_duty, self._dc_min)
            self.send_duty(left, right)
            return

        if self._final_approach_enabled:
            pose = self.get_pose_2d()
            if pose is None:
                self.stop()
                self.publish_status('FAILED')
                self._final_approach_enabled = False
                self.get_logger().warn('Final approach failed: no TF pose available (map->base_link).')
                return

            if self._final_target_id is None:
                self.stop()
                return

            self.request_final_target_pose()

            target_xy = self._final_target_xy
            if target_xy is None:
                timeout_s = float(self.get_parameter('final_target_timeout').value)
                last_seen = self._final_target_last_seen_wall
                if last_seen is None:
                    waiting_for_first_fix = (
                        self._final_target_request_pending
                        or (time.time() - self._final_target_request_last_wall) <= timeout_s
                    )
                    if waiting_for_first_fix:
                        self.stop()
                        return

                    self.stop()
                    self.publish_status('FAILED')
                    self._final_approach_enabled = False
                    self.get_logger().warn(
                        f'Final approach failed: no pose received yet for target {self._final_target_id}.'
                    )
                    return

                if (time.time() - last_seen) > timeout_s:
                    self.stop()
                    self.publish_status('FAILED')
                    self._final_approach_enabled = False
                    self.get_logger().warn(f'Final approach failed: target {self._final_target_id} timed out.')
                    return
                self.stop()
                return

            self._final_controller.update_gains(
                nominal_speed=float(self.get_parameter('final_nominal_speed').value),
                turn_gain=float(self.get_parameter('final_turn_gain').value),
                max_angular_speed=float(self.get_parameter('final_max_angular_speed').value),
                turn_in_place_yaw_thresh=float(self.get_parameter('final_turn_in_place_yaw_thresh').value),
                stop_distance=float(self.get_parameter('final_stop_distance').value),
            )
            command = self._final_controller.compute_command(pose, target_xy)
            if command.reached:
                self.stop()
                self.publish_status('REACHED')
                self._final_approach_enabled = False
                return

            left, right = self.enforce_motor_deadzone_pair(command.left, command.right, self._dc_min)
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
                w = clamp(k_turn * yaw_err, -wmax, wmax)

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
            k_turn = float(self.get_parameter('turn_gain').value)
            w = clamp(k_turn * yaw_err, -wmax, wmax)

            left, right = self.enforce_motor_deadzone_pair(-w, w, self._dc_min)
            self.send_duty(left, right)
            return

        # Pure Pursuit curvature: kappa = 2*y_r / L^2
        kappa = (2.0 * y_r) / (lookahead * lookahead)

        v_nom = float(self.get_parameter('nominal_linear_speed').value)
        wmax = float(self.get_parameter('max_angular_speed').value)
        k_steer = float(self.get_parameter('steering_gain').value)


        # Slow down in curves (simple, stable indoors)
        v_curve = v_nom / (1.0 + 3.0 * abs(kappa))
        v_curve = clamp(v_curve, 0.0, v_nom)

        # Slow down as we approach the final goal (improves accuracy / reduces overshoot)
        slow_radius = float(self.get_parameter('goal_slow_radius').value)
        v_min = float(self.get_parameter('min_linear_speed').value)

        slow_radius = max(0.05, slow_radius)
        v_min = max(self._dc_min + 0.02, min(v_min, v_nom))  # keep above deadzone margin

        approach = clamp(dist_to_goal / slow_radius, 0.0, 1.0)
        v_goal = v_min + (v_nom - v_min) * approach

        v = min(v_curve, v_goal)
        v = clamp(v, 0.0, v_nom)

        # Steering
        w = k_steer * v * kappa
        w = clamp(w, -wmax, wmax)

        # Deadband-aware feasibility: for forward motion, both wheels should stay >= min duty.
        # If v is small, cap steering so v-|w| does not fall into deadband.
        if v > 0.0:
            w_deadband_limit = max(0.0, v - self._dc_min)
            w = clamp(w, -w_deadband_limit, w_deadband_limit)

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
