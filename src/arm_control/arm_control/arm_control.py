from collections import deque
from dataclasses import dataclass
from enum import Enum

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from robp_interfaces.msg import ArmControl
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import String

from arm_control.arm_kinematics import DEFAULT_PICKUP_Z
from arm_control.arm_kinematics import DROP_POSE
from arm_control.arm_kinematics import HOLDING_POSE
from arm_control.arm_kinematics import IDLE_POSE
from arm_control.arm_kinematics import INITIAL_POSITION
from arm_control.arm_kinematics import START_POSITION
from arm_control.arm_kinematics import make_planar_target
from arm_control.arm_kinematics import planar_to_joint_target
from arm_control.arm_kinematics import rho_midpoint


MS_PER_DEGREE = 60
MIN_TIME_MS = 200
MAX_TIME_MS = 3000

OPEN_GRIPPER_ANGLE = 10.0
CLOSED_GRIPPER_ANGLE = 100.0
BASE_CENTER_ANGLE = 120.0
SAFE_START_WRIST_ANGLE = 120.0

VISION_TOPIC = '/arm/vision/green_cube_center'
ACTION_TOPIC = '/arm/action'
RESULT_TOPIC = '/arm/result'
CONTROL_TOPIC = '/arm/control'

CONTROL_RATE_HZ = 10.0
VISION_TIMEOUT_SEC = 1.0

TARGET_PIXEL_X = 320
TARGET_PIXEL_Y = 424
ALIGN_X_TOLERANCE = 12
ALIGN_Y_TOLERANCE = 10
PIXEL_TO_MM = 0.217
PIXEL_TO_ALPHA_DEG = 0.10
MAX_RHO_STEP_MM = 6.0
MAX_ALPHA_STEP_DEG = 2.0

REQUIRED_DETECTIONS = 3
STABLE_X_TOLERANCE = 8
STABLE_Y_TOLERANCE = 8


class State(Enum):
    START = 'start'
    START_PREPARE_WRIST = 'start_prepare_wrist'
    START_PREPARE_REST = 'start_prepare_rest'
    START_FINAL = 'start_final'
    MOVING_TO_IDLE = 'moving_to_idle'
    IDLE = 'idle'
    MOVING_TO_OBSERVE = 'moving_to_observe'
    ALIGNING = 'aligning'
    CLOSING_GRIPPER = 'closing_gripper'
    LIFTING = 'lifting'
    HOLDING = 'holding'
    MOVING_TO_DROP = 'moving_to_drop'
    OPENING_FOR_DROP = 'opening_for_drop'
    RETURNING_TO_IDLE = 'returning_to_idle'
    ERROR = 'error'


@dataclass
class VisionDetection:
    center_x: int
    center_y: int


class ArmControlNode(Node):
    def __init__(self):
        super().__init__('arm_control')

        self.state = State.START
        self.position = [float(value) for value in INITIAL_POSITION]
        self.new_position = self.position.copy()
        self.motion_complete_time = self.get_clock().now()

        self.current_target_rho = rho_midpoint()
        self.current_target_alpha = 0.0
        self.current_target_z = DEFAULT_PICKUP_Z

        self.latest_detection = None
        self.latest_detection_time = None
        self.detection_history = deque(maxlen=REQUIRED_DETECTIONS)

        self.control_pub = self.create_publisher(ArmControl, CONTROL_TOPIC, 10)
        self.result_pub = self.create_publisher(String, RESULT_TOPIC, 10)

        self.create_subscription(String, ACTION_TOPIC, self.action_callback, 10)
        self.create_subscription(Int32MultiArray, VISION_TOPIC, self.vision_callback, 10)

        self.control_timer = self.create_timer(1.0 / CONTROL_RATE_HZ, self.control_loop)

    def action_callback(self, msg: String):
        command = msg.data.strip().upper()
        if command == 'START':
            self.handle_start_command()
        elif command == 'PICK_UP':
            self.handle_pickup_command()
        elif command == 'DROP':
            self.handle_drop_command()

    def vision_callback(self, msg: Int32MultiArray):
        if len(msg.data) < 2:
            return

        detection = VisionDetection(center_x=int(msg.data[0]), center_y=int(msg.data[1]))
        self.latest_detection = detection
        self.latest_detection_time = self.get_clock().now()
        self.detection_history.append(detection)

    def control_loop(self):
        if self.is_motion_active():
            return

        if self.state == State.START_PREPARE_WRIST:
            self.command_start_prepare_rest()
            return

        if self.state == State.START_PREPARE_REST:
            self.command_start_final()
            return

        if self.state == State.START_FINAL:
            self.command_idle_pose(new_state=State.MOVING_TO_IDLE)
            return

        if self.state == State.MOVING_TO_IDLE:
            self.transition_to(State.IDLE)
            self.publish_result('START_SUCCESS')
            return

        if self.state == State.MOVING_TO_OBSERVE:
            self.transition_to(State.ALIGNING)
            return

        if self.state == State.ALIGNING:
            self.update_alignment()
            return

        if self.state == State.CLOSING_GRIPPER:
            self.command_named_pose(HOLDING_POSE, new_state=State.LIFTING)
            return

        if self.state == State.LIFTING:
            self.transition_to(State.HOLDING)
            self.publish_result('PICK_UP_SUCCESS')
            return

        if self.state == State.MOVING_TO_DROP:
            self.command_gripper(OPEN_GRIPPER_ANGLE, new_state=State.OPENING_FOR_DROP)
            return

        if self.state == State.OPENING_FOR_DROP:
            self.command_named_pose(IDLE_POSE, new_state=State.RETURNING_TO_IDLE)
            return

        if self.state == State.RETURNING_TO_IDLE:
            self.transition_to(State.IDLE)
            self.publish_result('DROP_SUCCESS')

    def handle_start_command(self):
        if self.state != State.START:
            self.publish_result('START_FAIL')
            return

        self.command_start_prepare_wrist()

    def handle_pickup_command(self):
        if self.state != State.IDLE:
            self.publish_result('PICK_UP_FAIL_NO_IDLE')
            return

        self.command_observe_pose()

    def handle_drop_command(self):
        if self.state != State.HOLDING:
            self.publish_result('DROP_FAIL_NO_OBJECT')
            return

        self.command_named_pose(DROP_POSE, new_state=State.MOVING_TO_DROP)

    def command_observe_pose(self):
        self.current_target_rho = rho_midpoint()
        self.current_target_alpha = 0.0
        self.current_target_z = DEFAULT_PICKUP_Z
        self.detection_history.clear()
        self.command_gripper(OPEN_GRIPPER_ANGLE)
        self.command_planar_target(
            rho=self.current_target_rho,
            alpha_deg=self.current_target_alpha,
            z=self.current_target_z,
            new_state=State.MOVING_TO_OBSERVE,
        )

    def command_idle_pose(self, new_state: State):
        idle_position = self.position.copy()
        idle_position[0] = OPEN_GRIPPER_ANGLE
        idle_position[2] = IDLE_POSE['p2']
        idle_position[3] = IDLE_POSE['p3']
        idle_position[4] = IDLE_POSE['p4']
        idle_position[5] = BASE_CENTER_ANGLE
        self.publish_arm_control(idle_position, new_state)

    def command_start_prepare_wrist(self):
        target_position = self.position.copy()
        target_position[2] = SAFE_START_WRIST_ANGLE
        self.publish_arm_control(target_position, State.START_PREPARE_WRIST)

    def command_start_prepare_rest(self):
        target_position = self.position.copy()
        target_position[0] = START_POSITION[0]
        target_position[1] = START_POSITION[1]
        target_position[3] = START_POSITION[3]
        target_position[4] = START_POSITION[4]
        target_position[5] = START_POSITION[5]
        self.publish_arm_control(target_position, State.START_PREPARE_REST)

    def command_start_final(self):
        target_position = self.position.copy()
        target_position[2] = START_POSITION[2]
        self.publish_arm_control(target_position, State.START_FINAL)

    def command_named_pose(self, pose: dict, new_state: State):
        target_position = self.position.copy()
        if 'p2' in pose:
            target_position[2] = pose['p2']
        if 'p3' in pose:
            target_position[3] = pose['p3']
        if 'p4' in pose:
            target_position[4] = pose['p4']
        if 'p5' in pose:
            target_position[5] = pose['p5']
        self.publish_arm_control(target_position, new_state)

    def command_gripper(self, angle: float, new_state: State | None = None):
        target_position = self.position.copy()
        target_position[0] = angle
        self.publish_arm_control(target_position, new_state)

    def command_planar_target(self, rho: float, alpha_deg: float, z: float, new_state: State | None = None):
        planar_target = make_planar_target(rho=rho, alpha_deg=alpha_deg, z=z)
        joint_target = planar_to_joint_target(planar_target)

        target_position = self.position.copy()
        target_position[2] = joint_target.wrist
        target_position[3] = joint_target.elbow
        target_position[4] = joint_target.shoulder
        target_position[5] = joint_target.base

        self.current_target_rho = planar_target.rho
        self.current_target_alpha = planar_target.alpha_deg
        self.current_target_z = planar_target.z
        self.publish_arm_control(target_position, new_state)

    def update_alignment(self):
        detection = self.get_stable_detection()
        if detection is None:
            if self.vision_is_stale():
                self.transition_to(State.IDLE)
                self.publish_result('PICK_UP_FAIL_TIMEOUT')
            return

        error_x = TARGET_PIXEL_X - detection.center_x
        error_y = TARGET_PIXEL_Y - detection.center_y

        if abs(error_x) <= ALIGN_X_TOLERANCE and abs(error_y) <= ALIGN_Y_TOLERANCE:
            self.command_gripper(CLOSED_GRIPPER_ANGLE, new_state=State.CLOSING_GRIPPER)
            return

        delta_rho = self.clamp_step(error_y * PIXEL_TO_MM, MAX_RHO_STEP_MM)
        delta_alpha = self.clamp_step(error_x * PIXEL_TO_ALPHA_DEG, MAX_ALPHA_STEP_DEG)

        try:
            self.command_planar_target(
                rho=self.current_target_rho + delta_rho,
                alpha_deg=self.current_target_alpha + delta_alpha,
                z=self.current_target_z,
                new_state=State.ALIGNING,
            )
        except ValueError as exc:
            self.transition_to(State.ERROR)
            self.publish_result(f'PICK_UP_FAIL_RANGE: {exc}')

    def publish_arm_control(self, target_position: list[float], new_state: State | None = None):
        self.new_position = [float(value) for value in target_position]

        msg = ArmControl()
        msg.position = self.new_position
        msg.time = self.compute_move_times(self.position, self.new_position)
        self.control_pub.publish(msg)

        self.position = self.new_position.copy()
        max_move_time_ms = max(msg.time) if msg.time else MIN_TIME_MS
        self.motion_complete_time = self.get_clock().now() + Duration(seconds=max_move_time_ms / 1000.0)

        if new_state is not None:
            self.transition_to(new_state)

    def compute_move_times(self, current_position: list[float], target_position: list[float]) -> list[int]:
        move_times = []
        for current, target in zip(current_position, target_position):
            diff = abs(target - current)
            move_time = int(diff * MS_PER_DEGREE)
            move_time = max(MIN_TIME_MS, min(MAX_TIME_MS, move_time))
            move_times.append(move_time)
        return move_times

    def get_stable_detection(self):
        if len(self.detection_history) < REQUIRED_DETECTIONS:
            return None

        xs = [d.center_x for d in self.detection_history]
        ys = [d.center_y for d in self.detection_history]

        if max(xs) - min(xs) > STABLE_X_TOLERANCE:
            return None
        if max(ys) - min(ys) > STABLE_Y_TOLERANCE:
            return None

        return VisionDetection(
            center_x=sum(xs) // len(xs),
            center_y=sum(ys) // len(ys),
        )

    def vision_is_stale(self) -> bool:
        if self.latest_detection_time is None:
            return True
        age = self.get_clock().now() - self.latest_detection_time
        return age > Duration(seconds=VISION_TIMEOUT_SEC)

    def is_motion_active(self) -> bool:
        return self.get_clock().now() < self.motion_complete_time

    def clamp_step(self, value: float, limit: float) -> float:
        return max(-limit, min(limit, value))

    def transition_to(self, new_state: State):
        self.state = new_state
        self.get_logger().info(f'Arm state -> {new_state.value}')

    def publish_result(self, text: str):
        msg = String()
        msg.data = text
        self.result_pub.publish(msg)
        self.get_logger().info(f'Arm result: {text}')


def main():
    rclpy.init()
    node = ArmControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
