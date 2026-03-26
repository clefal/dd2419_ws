#!/usr/bin/env python3

import csv
import os
from datetime import datetime
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node

from robp_interfaces.msg import DutyCycles


class CallibrationHelper(Node):
    def __init__(self) -> None:
        super().__init__('callibration_helper')

        self.declare_parameter('command_topic', '/phidgets/motor/duty_cycles')
        self.declare_parameter('measured_topic', '/phidgets/motor/current_duty_cycles')
        self.declare_parameter('output_path', '')
        self.declare_parameter('flush_period', 1.0)

        command_topic = str(self.get_parameter('command_topic').value)
        measured_topic = str(self.get_parameter('measured_topic').value)
        flush_period = max(0.1, float(self.get_parameter('flush_period').value))

        self._output_path = self._resolve_output_path(str(self.get_parameter('output_path').value))
        os.makedirs(os.path.dirname(self._output_path), exist_ok=True)
        self._csv_file = open(self._output_path, 'w', newline='', encoding='ascii')
        self._writer = csv.writer(self._csv_file)
        self._writer.writerow([
            'event_time_sec',
            'source',
            'cmd_left',
            'cmd_right',
            'measured_left',
            'measured_right',
        ])

        self._latest_cmd: Optional[Tuple[float, float]] = None
        self._latest_measured: Optional[Tuple[float, float]] = None
        self._rows_written = 0

        self.create_subscription(DutyCycles, command_topic, self.command_callback, 50)
        self.create_subscription(DutyCycles, measured_topic, self.measured_callback, 50)
        self._flush_timer = self.create_timer(flush_period, self.flush_csv)

        self.get_logger().info(f'Logging commanded duties from {command_topic}')
        self.get_logger().info(f'Logging measured duties from {measured_topic}')
        self.get_logger().info(f'Writing CSV to {self._output_path}')

    def _resolve_output_path(self, configured_path: str) -> str:
        if configured_path.strip():
            return os.path.abspath(configured_path)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        package_dir = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.join(package_dir, 'logs')
        return os.path.join(logs_dir, f'wheel_duty_log_{timestamp}.csv')

    def command_callback(self, msg: DutyCycles) -> None:
        self._latest_cmd = (float(msg.duty_cycle_left), float(msg.duty_cycle_right))
        self.write_row('command')

    def measured_callback(self, msg: DutyCycles) -> None:
        self._latest_measured = (float(msg.duty_cycle_left), float(msg.duty_cycle_right))
        self.write_row('measured')

    def write_row(self, source: str) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        cmd_left, cmd_right = self._latest_cmd if self._latest_cmd is not None else ('', '')
        measured_left, measured_right = (
            self._latest_measured if self._latest_measured is not None else ('', '')
        )
        self._writer.writerow([
            f'{now:.6f}',
            source,
            cmd_left,
            cmd_right,
            measured_left,
            measured_right,
        ])
        self._rows_written += 1

    def flush_csv(self) -> None:
        self._csv_file.flush()

    def close_csv(self) -> None:
        try:
            self._csv_file.flush()
            self._csv_file.close()
        except Exception:
            pass


def main() -> None:
    rclpy.init()
    node = CallibrationHelper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(
            f'Stopping callibration_helper after writing {node._rows_written} rows to {node._output_path}'
        )
        node.close_csv()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
