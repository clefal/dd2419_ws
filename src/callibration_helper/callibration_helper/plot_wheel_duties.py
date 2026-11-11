#!/usr/bin/env python3

import csv
import os
from typing import List, Optional

import matplotlib.pyplot as plt


# Set this to the CSV file you want to inspect.
CSV_PATH = r'c:\programmierstuff\dd2419_ws\src\callibration_helper\callibration_helper\logs\wheel_duty_log_YYYYMMDD_HHMMSS.csv'


def parse_optional_float(value: str) -> Optional[float]:
    value = value.strip()
    if value == '':
        return None
    return float(value)


def load_csv(path: str):
    times: List[float] = []
    cmd_left: List[Optional[float]] = []
    cmd_right: List[Optional[float]] = []
    measured_left: List[Optional[float]] = []
    measured_right: List[Optional[float]] = []

    with open(path, 'r', newline='', encoding='ascii') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            times.append(float(row['event_time_sec']))
            cmd_left.append(parse_optional_float(row['cmd_left']))
            cmd_right.append(parse_optional_float(row['cmd_right']))
            measured_left.append(parse_optional_float(row['measured_left']))
            measured_right.append(parse_optional_float(row['measured_right']))

    if not times:
        raise ValueError(f'CSV file is empty: {path}')

    t0 = times[0]
    rel_time = [t - t0 for t in times]
    return rel_time, cmd_left, cmd_right, measured_left, measured_right


def main() -> None:
    csv_path = os.path.abspath(CSV_PATH)
    rel_time, cmd_left, cmd_right, measured_left, measured_right = load_csv(csv_path)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].plot(rel_time, cmd_left, label='cmd_left', linewidth=1.8)
    axes[0].plot(rel_time, measured_left, label='measured_left', linewidth=1.4, alpha=0.85)
    axes[0].set_ylabel('Left duty')
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(rel_time, cmd_right, label='cmd_right', linewidth=1.8)
    axes[1].plot(rel_time, measured_right, label='measured_right', linewidth=1.4, alpha=0.85)
    axes[1].set_ylabel('Right duty')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(csv_path)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
