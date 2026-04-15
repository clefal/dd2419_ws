#!/usr/bin/env python3
import csv
import math
from typing import List, Tuple, Dict

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Point, TransformStamped, PolygonStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


def yaw_to_quat(yaw_rad: float) -> Tuple[float, float, float, float]:
    """Yaw-only quaternion (x,y,z,w)."""
    half = 0.5 * yaw_rad
    return 0.0, 0.0, math.sin(half), math.cos(half)


def unit_scale_to_meters(units: str) -> float:
    """
    Convert from input units to meters.
    """
    u = units.strip().lower()
    if u in ("m", "meter", "meters"):
        return 1.0
    if u in ("cm", "centimeter", "centimeters"):
        return 0.01
    if u in ("mm", "millimeter", "millimeters"):
        return 0.001
    raise ValueError(f"Unknown units '{units}'. Use: m, cm, or mm.")


def read_workspace_csv(path: str, scale: float) -> List[Tuple[float, float]]:
    """
    format:
      x, y
    """
    pts: List[Tuple[float, float]] = []
    print('Reading workspace from', path)
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = [fn.strip() for fn in (reader.fieldnames or [])]
        for row in reader:
            if not row:
                continue
            # find x,y keys
            x_val = None
            y_val = None
            for k, v in row.items():
                if k is None:
                    continue
                kk = k.strip().lower()
                if kk == "x":
                    x_val = v
                elif kk == "y":
                    y_val = v
            if x_val is None or y_val is None:
                continue
            x = float(str(x_val).strip()) * scale
            y = float(str(y_val).strip()) * scale
            pts.append((x, y))

    if len(pts) < 3:
        raise ValueError(f"Workspace polygon needs >= 3 points, got {len(pts)} from {path}")
    return pts


def read_map_csv(path: str, scale: float):
    """
    Returns: start (x,y,deg), objects list, box (x,y,deg or None)
    """
    start = None
    box = None
    objects: List[Tuple[float, float, float]] = []

    with open(path, "r", newline="") as f:
        reader = csv.DictReader((line.replace("\ufeff", "") for line in f))
        for row in reader:
            if not row:
                continue

            # Normalize keys
            data: Dict[str, str] = {k.strip().lower(): (v.strip() if v is not None else "") for k, v in row.items() if k}

            t = data.get("type", "").upper()
            if not t:
                continue

            x = float(data.get("x", "0")) * scale
            y = float(data.get("y", "0")) * scale
            ang_deg = float(data.get("angle", "0") or "0")

            if t == "S":
                start = (x, y, ang_deg)
            elif t == "O":
                objects.append((x, y, ang_deg))
            elif t == "B":
                box = (x, y, ang_deg)

    if start is None:
        raise ValueError(f"No start 'S' found in {path}")

    return start, objects, box


class WorkspaceAndFrames(Node):
    def __init__(self):
        super().__init__("workspace_and_frames")

        # Params
        self.declare_parameter("workspace_csv", "")
        self.declare_parameter("map_csv", "")
        self.declare_parameter("input_units", "cm")
        self.declare_parameter("frame_map", "map")
        self.declare_parameter("frame_odom", "odom")
        self.declare_parameter("object_frame_prefix", "object")
        self.declare_parameter("box_frame", "box")
        self.declare_parameter("workspace_polygon_topic", "/workspace")
        self.declare_parameter("line_width", 0.05)
        self.declare_parameter("publish_odom_frames", True)

        workspace_csv = self.get_parameter("workspace_csv").value
        map_csv = self.get_parameter("map_csv").value
        units = self.get_parameter("input_units").value

        if not workspace_csv:
            raise RuntimeError("Parameter 'workspace_csv' is required.")
        if not map_csv:
            raise RuntimeError("Parameter 'map_csv' is required.")

        self.frame_map = self.get_parameter("frame_map").value
        self.frame_odom = self.get_parameter("frame_odom").value
        self.object_prefix = self.get_parameter("object_frame_prefix").value
        self.box_frame = self.get_parameter("box_frame").value
        self.workspace_polygon_topic = self.get_parameter("workspace_polygon_topic").value
        self.line_width = float(self.get_parameter("line_width").value)
        self.publish_odom_frames = self.get_parameter("publish_odom_frames").value

        scale = unit_scale_to_meters(units)

        # Publisher (latched / transient)
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,    
        )
        self.polygon_pub = self.create_publisher(PolygonStamped, self.workspace_polygon_topic, qos)

        # Static TF broadcaster
        self.static_broadcaster = StaticTransformBroadcaster(self)

        # Read files
        polygon_pts = read_workspace_csv(workspace_csv, scale)
        start, objects, box = read_map_csv(map_csv, scale)

        # Publish & broadcast
        self.publish_workspace_polygon(polygon_pts)
        self.broadcast_static_transforms(start, objects, box)

        self.get_logger().info(
            f"OK. Workspace points={len(polygon_pts)}, objects={len(objects)}, box={'yes' if box else 'no'}. "
            f"Units={units} (scale={scale})."
        )

    def broadcast_static_transforms(self, start, objects, box):
        now = self.get_clock().now().to_msg()
        tfs: List[TransformStamped] = []

        sx, sy, sdeg = start
        qx, qy, qz, qw = yaw_to_quat(math.radians(sdeg))

        if self.publish_odom_frames:
            # map -> odom at start pose
            tf_map_odom = TransformStamped()
            tf_map_odom.header.stamp = now
            tf_map_odom.header.frame_id = self.frame_map
            tf_map_odom.child_frame_id = self.frame_odom
            tf_map_odom.transform.translation.x = float(sx)
            tf_map_odom.transform.translation.y = float(sy)
            tf_map_odom.transform.translation.z = 0.0
            tf_map_odom.transform.rotation.x = qx
            tf_map_odom.transform.rotation.y = qy
            tf_map_odom.transform.rotation.z = qz
            tf_map_odom.transform.rotation.w = qw
            tfs.append(tf_map_odom)


        # map -> start
        tf_map_start = TransformStamped()
        tf_map_start.header.stamp = now
        tf_map_start.header.frame_id = self.frame_map
        tf_map_start.child_frame_id = "start"
        tf_map_start.transform.translation.x = float(sx)
        tf_map_start.transform.translation.y = float(sy)
        tf_map_start.transform.translation.z = 0.0
        tf_map_start.transform.rotation.x = qx
        tf_map_start.transform.rotation.y = qy
        tf_map_start.transform.rotation.z = qz
        tf_map_start.transform.rotation.w = qw
        tfs.append(tf_map_start)

        if self.publish_odom_frames:
            tf_map_odom_temp = TransformStamped()
            tf_map_odom_temp.header.stamp = now
            tf_map_odom_temp.header.frame_id = self.frame_map
            tf_map_odom_temp.child_frame_id = "odom_temp"
            tf_map_odom_temp.transform.translation.x = float(sx)
            tf_map_odom_temp.transform.translation.y = float(sy)
            tf_map_odom_temp.transform.translation.z = 0.0
            tf_map_odom_temp.transform.rotation.x = qx
            tf_map_odom_temp.transform.rotation.y = qy
            tf_map_odom_temp.transform.rotation.z = qz
            tf_map_odom_temp.transform.rotation.w = qw
            tfs.append(tf_map_odom_temp)

        # map -> objectN
        for i, (ox, oy, odeg) in enumerate(objects):
            qx, qy, qz, qw = yaw_to_quat(math.radians(odeg))
            tf_obj = TransformStamped()
            tf_obj.header.stamp = now
            tf_obj.header.frame_id = self.frame_map
            tf_obj.child_frame_id = f"{self.object_prefix}{i}"
            tf_obj.transform.translation.x = float(ox)
            tf_obj.transform.translation.y = float(oy)
            tf_obj.transform.translation.z = 0.0
            tf_obj.transform.rotation.x = qx
            tf_obj.transform.rotation.y = qy
            tf_obj.transform.rotation.z = qz
            tf_obj.transform.rotation.w = qw
            tfs.append(tf_obj)

        # map -> box
        if box is not None:
            bx, by, bdeg = box
            qx, qy, qz, qw = yaw_to_quat(math.radians(bdeg))
            tf_box = TransformStamped()
            tf_box.header.stamp = now
            tf_box.header.frame_id = self.frame_map
            tf_box.child_frame_id = self.box_frame
            tf_box.transform.translation.x = float(bx)
            tf_box.transform.translation.y = float(by)
            tf_box.transform.translation.z = 0.0
            tf_box.transform.rotation.x = qx
            tf_box.transform.rotation.y = qy
            tf_box.transform.rotation.z = qz
            tf_box.transform.rotation.w = qw
            tfs.append(tf_box)

        self.static_broadcaster.sendTransform(tfs)

    def publish_workspace_polygon(self, pts_xy: List[Tuple[float, float]]):
        poly = PolygonStamped()
        poly.header.frame_id = self.frame_map
        poly.header.stamp = self.get_clock().now().to_msg()

        for x, y in pts_xy:
            poly.polygon.points.append(Point(x=float(x), y=float(y), z=0.0))

        self.polygon_pub.publish(poly)


def main():
    rclpy.init()
    node = WorkspaceAndFrames()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
