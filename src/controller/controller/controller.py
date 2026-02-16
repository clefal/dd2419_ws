#!/usr/bin/env python3

import sys
import threading
import math

import rclpy
from rclpy.node import Node

from robp_interfaces.msg import Encoders, DutyCycles


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class CalibrationHelper(Node):
    def __init__(self):
        super().__init__('calibration_helper')

        self._ticks_per_rev = 48 * 64
        self._recording = False
        self._sum_left = 0
        self._sum_right = 0
        self._progress_left = 0.0
        self._progress_right = 0.0
        self._last_radius = None
        self._auto_target_ticks = None
        self._auto_drive_sign = 1.0

        self._cmd_pub = self.create_publisher(DutyCycles, '/phidgets/motor/duty_cycles', 10)
        self.create_subscription(Encoders, '/phidgets/motor/encoders', self.encoder_callback, 10)

        self._thread = threading.Thread(target=self.input_loop, daemon=True)
        self._thread.start()

        self.print_help()

    def encoder_callback(self, msg: Encoders):
        if not self._recording:
            return
        dl = int(msg.delta_encoder_left)
        dr = int(msg.delta_encoder_right)
        self._sum_left += dl
        self._sum_right += dr

        if self._auto_target_ticks is not None:
            # Count only progress in commanded direction to avoid cancellation by jitter/slip.
            self._progress_left += max(0.0, self._auto_drive_sign * dl)
            self._progress_right += max(0.0, self._auto_drive_sign * dr)

        if self._auto_target_ticks is not None and self.avg_progress_ticks() >= self._auto_target_ticks:
            self.stop_motors()
            self._recording = False
            self.get_logger().info(
                f'auto drive complete at progress_ticks={self.avg_progress_ticks():.1f}, '
                f'net_ticks={self.avg_abs_ticks():.1f}. '
                'Use calc_r <distance_m>.'
            )
            self._auto_target_ticks = None

    def send_duty(self, left, right):
        m = DutyCycles()
        m.duty_cycle_left = float(clamp(left, -1.0, 1.0))
        m.duty_cycle_right = float(clamp(right, -1.0, 1.0))
        self._cmd_pub.publish(m)

    def stop_motors(self):
        self.send_duty(0.0, 0.0)

    def reset_recording(self):
        self._sum_left = 0
        self._sum_right = 0
        self._progress_left = 0.0
        self._progress_right = 0.0

    def avg_abs_ticks(self):
        return 0.5 * (abs(self._sum_left) + abs(self._sum_right))

    def avg_progress_ticks(self):
        return 0.5 * (self._progress_left + self._progress_right)

    def print_status(self):
        self.get_logger().info(
            f'recording={self._recording} left_ticks={self._sum_left} right_ticks={self._sum_right}'
        )

    def print_help(self):
        help_text = """
Calibration helper commands:
  help
  status
  reset
  record on|off
  stop
  drive_turns <duty> <wheel_turns>
  spin <duty>         # in-place spin: left=-duty, right=+duty
  calc_r <distance_m>
  calc_b <turns> <radius_m>
  calc_b_last_r <turns>
  quit

Alternative radius test (fixed wheel turns, then tape-measure):
  drive_turns 0.18 10
  (helper auto-stops after avg wheel turns = 10)
  calc_r <measured_distance_m>

Typical base test:
  reset
  record on
  spin 0.16
  (do n full turns and align heading mark)
  stop
  record off
  calc_b_last_r 10
"""
        self.get_logger().info(help_text)

    def input_loop(self):
        while rclpy.ok():
            line = sys.stdin.readline()
            if not line:
                continue

            cmd = line.strip().split()
            if not cmd:
                continue

            try:
                op = cmd[0].lower()
                if op == 'help':
                    self.print_help()
                elif op == 'status':
                    self.print_status()
                elif op == 'reset':
                    self.reset_recording()
                    self.get_logger().info('tick sums reset')
                elif op == 'record':
                    if len(cmd) != 2 or cmd[1] not in ('on', 'off'):
                        self.get_logger().info('usage: record on|off')
                        continue
                    self._recording = (cmd[1] == 'on')
                    self.get_logger().info(f'recording={self._recording}')
                elif op == 'stop':
                    self.stop_motors()
                    self._auto_target_ticks = None
                    self.get_logger().info('motors stopped')
                elif op == 'drive_turns':
                    if len(cmd) != 3:
                        self.get_logger().info('usage: drive_turns <duty> <wheel_turns>')
                        continue
                    duty = float(cmd[1])
                    wheel_turns = float(cmd[2])
                    if wheel_turns <= 0.0:
                        self.get_logger().info('wheel_turns must be > 0')
                        continue
                    self.reset_recording()
                    self._recording = True
                    self._auto_target_ticks = wheel_turns * self._ticks_per_rev
                    self._auto_drive_sign = 1.0 if duty >= 0.0 else -1.0
                    self.send_duty(duty, duty)
                    self.get_logger().info(
                        f'auto drive started: duty={duty:.3f}, turns={wheel_turns:.3f}'
                    )
                elif op == 'spin':
                    if len(cmd) != 2:
                        self.get_logger().info('usage: spin <duty>')
                        continue
                    duty = float(cmd[1])
                    self._auto_target_ticks = None
                    self.send_duty(-duty, duty)
                elif op == 'calc_r':
                    if len(cmd) != 2:
                        self.get_logger().info('usage: calc_r <distance_m>')
                        continue
                    distance_m = float(cmd[1])
                    n_avg = self.avg_abs_ticks()
                    if n_avg <= 0.0:
                        self.get_logger().info('no recorded ticks')
                        continue
                    radius = distance_m * self._ticks_per_rev / (2.0 * math.pi * n_avg)
                    self._last_radius = radius
                    self.get_logger().info(f'estimated wheel radius r={radius:.6f} m')
                elif op == 'calc_b':
                    if len(cmd) != 3:
                        self.get_logger().info('usage: calc_b <turns> <radius_m>')
                        continue
                    turns = float(cmd[1])
                    radius = float(cmd[2])
                    if turns <= 0.0:
                        self.get_logger().info('turns must be > 0')
                        continue
                    tick_sum = abs(self._sum_left) + abs(self._sum_right)
                    base = radius * tick_sum / (turns * self._ticks_per_rev)
                    self.get_logger().info(f'estimated wheel base b={base:.6f} m')
                elif op == 'calc_b_last_r':
                    if len(cmd) != 2:
                        self.get_logger().info('usage: calc_b_last_r <turns>')
                        continue
                    if self._last_radius is None:
                        self.get_logger().info('no saved radius yet, run calc_r first')
                        continue
                    turns = float(cmd[1])
                    if turns <= 0.0:
                        self.get_logger().info('turns must be > 0')
                        continue
                    tick_sum = abs(self._sum_left) + abs(self._sum_right)
                    base = self._last_radius * tick_sum / (turns * self._ticks_per_rev)
                    self.get_logger().info(f'estimated wheel base b={base:.6f} m')
                elif op == 'quit':
                    self.stop_motors()
                    self._auto_target_ticks = None
                    rclpy.shutdown()
                    return
                else:
                    self.get_logger().info(f'unknown command: {op}')
            except ValueError:
                self.get_logger().info('invalid numeric input')
            except Exception as e:
                self.get_logger().error(f'command error: {e}')


def main():
    rclpy.init()
    node = CalibrationHelper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.stop_motors()
        except Exception:
            pass
    rclpy.shutdown()


if __name__ == '__main__':
    main()
