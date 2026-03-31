from collections import deque
from dataclasses import dataclass
from enum import Enum

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from robp_interfaces.msg import ArmControl
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import String

from arm_control.arm_kinematics import (
    MAX_RHO,
    DEFAULT_PICKUP_Z,
    IDLE_Z,
    DROP_POSE,
    LIFTING_POSE,
    HOLDING_POSE,
    IDLE_POSE,
    INITIAL_POSITION,
    START_SAFE_POSITION,
    BASE_MIN_RHO,
    get_min_rho,
    make_planar_target,
    planar_to_joint_target,
    rho_midpoint,
)


MS_PER_DEGREE = 60
MIN_TIME_MS = 200
MAX_TIME_MS = 3000

OPEN_GRIPPER_ANGLE = 10.0
CLOSED_GRIPPER_ANGLE = 105.0
BASE_CENTER_ANGLE = 120.0

VISION_TOPIC = '/arm/vision/green_cube_center'
ACTION_TOPIC = '/arm/action'
RESULT_TOPIC = '/arm/result'
CONTROL_TOPIC = '/arm/control'

CONTROL_RATE_HZ = 10.0
VISION_TIMEOUT_SEC = 1.0

TARGET_PIXEL_X = 300
TARGET_PIXEL_Y = 400
ALIGN_X_TOLERANCE = 25
ALIGN_Y_TOLERANCE = 10
PIXEL_TO_MM = 0.15
PIXEL_TO_ALPHA_DEG = 0.055
MAX_RHO_STEP_MM = 6.0
MAX_ALPHA_STEP_DEG = 2.0

DESCENT_STEP_MM = 5.0
FINAL_PICKUP_Z = DEFAULT_PICKUP_Z   #  current low value
START_PICKUP_Z = IDLE_Z - 50.0  # higher starting point
ALIGNMENT_Z = FINAL_PICKUP_Z + 10.0  # stop aligning below this Z to avoid vision issues
REQUIRED_DETECTIONS = 3
STABLE_X_TOLERANCE = 8
STABLE_Y_TOLERANCE = 8

TEST_RHO_STEP_MM = 5.0
TEST_ALPHA_STEP_DEG = 5.0
TEST_Z_STEP_MM = 5.0
DEBUG_VISION_UPDATES = True
VISION_LOG_MIN_INTERVAL_SEC = 0.75
VISION_LOG_DELTA_PIXELS = 10
OUT_OF_REACH_CONFIRMATION_STEPS = 3
OUT_OF_REACH_MIN_ERROR_IMPROVEMENT = 12.0


class State(Enum):
    START = 'start'
    MOVING_TO_START_SAFE = 'moving_to_start_safe'
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
        self.last_logged_detection = None
        self.last_vision_log_time = None
        self.track_only_mode = False
        self.out_of_reach_counter = 0
        self.alignment_error_history = deque(maxlen=OUT_OF_REACH_CONFIRMATION_STEPS + 1)

        self.control_pub = self.create_publisher(ArmControl, CONTROL_TOPIC, 10)
        self.result_pub = self.create_publisher(String, RESULT_TOPIC, 10)

        self.create_subscription(String, ACTION_TOPIC, self.action_callback, 10)
        self.create_subscription(Int32MultiArray, VISION_TOPIC, self.vision_callback, 10)

        self.control_timer = self.create_timer(1.0 / CONTROL_RATE_HZ, self.control_loop)

    def action_callback(self, msg: String):
        command = msg.data.strip().upper()
        if command == 'START':
            self.handle_start_command()
        elif command == 'TRACK_ONLY':
            self.handle_track_only_command()
        elif command == 'PICK_UP':
            self.handle_pickup_command()
        elif command == 'DROP':
            self.handle_drop_command()
        elif command == 'TEST_CENTER_LOW':
            self.handle_test_planar_command(
                rho=175,
                alpha_deg=0.0,
                z=FINAL_PICKUP_Z,
                label='TEST_CENTER_LOW',
            )
        elif command == 'TEST_CENTER_HIGH':
            self.handle_test_planar_command(
                rho=175,
                alpha_deg=0.0,
                z=START_PICKUP_Z,
                label='TEST_CENTER_HIGH',
            )
        elif command == 'TEST_RHO_IN':
            self.handle_test_planar_command(
                rho=self.current_target_rho - TEST_RHO_STEP_MM,
                alpha_deg=self.current_target_alpha,
                label='TEST_RHO_IN',
            )
        elif command == 'TEST_RHO_OUT':
            self.handle_test_planar_command(
                rho=self.current_target_rho + TEST_RHO_STEP_MM,
                alpha_deg=self.current_target_alpha,
                label='TEST_RHO_OUT',
            )
        elif command == 'TEST_LEFT':
            self.handle_test_planar_command(
                rho=self.current_target_rho,
                alpha_deg=self.current_target_alpha + TEST_ALPHA_STEP_DEG,
                label='TEST_LEFT',
            )
        elif command == 'TEST_RIGHT':
            self.handle_test_planar_command(
                rho=self.current_target_rho,
                alpha_deg=self.current_target_alpha - TEST_ALPHA_STEP_DEG,
                label='TEST_RIGHT',
            )
        elif command == 'TEST_Z_UP':
            self.handle_test_planar_command(
                rho=self.current_target_rho,
                alpha_deg=self.current_target_alpha,
                z=self.current_target_z + TEST_Z_STEP_MM,
                label='TEST_Z_UP',
            )
        elif command == 'TEST_Z_DOWN':
            self.handle_test_planar_command(
                rho=self.current_target_rho,
                alpha_deg=self.current_target_alpha,
                z=self.current_target_z - TEST_Z_STEP_MM,
                label='TEST_Z_DOWN',
            )
        elif command == 'TEST_STATUS':
            self.publish_result(
                'TEST_STATUS '
                f'rho={self.current_target_rho:.1f} '
                f'alpha={self.current_target_alpha:.1f} '
                f'z={self.current_target_z:.1f}'
            )
        elif command == 'TEST_CLOSE_GRIPPER':
            closed_gripper_pose = self.position.copy()
            closed_gripper_pose[0] = CLOSED_GRIPPER_ANGLE
      
            self.publish_arm_control(closed_gripper_pose, new_state=None)
            #self.command_named_pose(HOLDING_POSE, new_state=State.LIFTING)



    def vision_callback(self, msg: Int32MultiArray):
        if len(msg.data) < 2:
            return

        detection = VisionDetection(center_x=int(msg.data[0]), center_y=int(msg.data[1]))
        self.latest_detection = detection
        self.latest_detection_time = self.get_clock().now()
        self.detection_history.append(detection)
        if DEBUG_VISION_UPDATES and self.should_log_detection(detection):
            self.get_logger().info(
                f'Vision update: x={detection.center_x} y={detection.center_y} '
                f'rho={self.current_target_rho:.1f} alpha={self.current_target_alpha:.1f} '
                f'z={self.current_target_z:.1f}'
            )
            self.last_logged_detection = detection
            self.last_vision_log_time = self.latest_detection_time

    def control_loop(self):
        if self.is_motion_active():
            return
        
        if self.state == State.MOVING_TO_START_SAFE:
            self.command_idle_pose(State.MOVING_TO_IDLE)
            return

        if self.state == State.MOVING_TO_IDLE:
            self.transition_to(State.IDLE)
            self.publish_result('IDLE_SUCCESS')
            return

        if self.state == State.MOVING_TO_OBSERVE:
            self.transition_to(State.ALIGNING)
            return

        if self.state == State.ALIGNING:
            self.update_alignment()
            return
        
        if self.state == State.CLOSING_GRIPPER:
            self.command_named_pose(LIFTING_POSE, new_state=State.LIFTING)
            return

        if self.state == State.LIFTING:
            #TODO check if cube is actually lifted by checking vision before holding pose, declaring failure if not lifted
            self.publish_result('PICK_UP_SUCCESS')
            self.command_named_pose(HOLDING_POSE, new_state=State.HOLDING)
            return
        
        if self.state == State.MOVING_TO_DROP:
            self.command_gripper(OPEN_GRIPPER_ANGLE, new_state=State.OPENING_FOR_DROP)
            return

        if self.state == State.OPENING_FOR_DROP:
            self.command_idle_pose(State.RETURNING_TO_IDLE)
            return

        if self.state == State.RETURNING_TO_IDLE:
            self.transition_to(State.IDLE)
            self.publish_result('DROP_SUCCESS')

    def handle_start_command(self):
        if self.state != State.START or self.is_motion_active():
            self.publish_result('START_FAIL')
            return

        target_position = START_SAFE_POSITION.copy()
        self.publish_arm_control(target_position, new_state=State.MOVING_TO_START_SAFE)


    def handle_pickup_command(self):
        if self.state != State.IDLE:
            self.publish_result('PICK_UP_FAIL_NO_IDLE')
            return

        self.track_only_mode = False
        self.command_observe_pose()

    def handle_track_only_command(self):
        if self.state != State.IDLE:
            self.publish_result('TRACK_ONLY_FAIL_NO_IDLE')
            return

        self.track_only_mode = True
        self.command_observe_pose()

    def handle_drop_command(self):
        if self.state != State.HOLDING:
            self.publish_result('DROP_FAIL_NO_OBJECT')
            return

        self.command_named_pose(DROP_POSE, new_state=State.MOVING_TO_DROP)

    def handle_test_planar_command(
        self,
        rho: float,
        alpha_deg: float,
        label: str,
        z: float | None = None,
    ):
        if self.state != State.IDLE:
            self.publish_result(f'{label}_FAIL_NO_IDLE')
            return

        test_z = self.current_target_z if z is None else z

        try:
            planar_target = make_planar_target(
                rho=rho,
                alpha_deg=alpha_deg,
                z=test_z,
            )
            joint_target = planar_to_joint_target(planar_target)
            self.get_logger().info(
                f'{label}: commanding rho={planar_target.rho:.1f} '
                f'alpha={planar_target.alpha_deg:.1f} z={planar_target.z:.1f} '
                f'-> p2={joint_target.wrist:.1f} p3={joint_target.elbow:.1f} '
                f'p4={joint_target.shoulder:.1f} p5={joint_target.base:.1f}'
            )
            self.command_planar_target(
                rho=rho,
                alpha_deg=alpha_deg,
                z=test_z,
            )
        except ValueError as exc:
            self.publish_result(f'{label}_FAIL {exc}')
            return

        self.publish_result(
            f'{label}_OK '
            f'rho={self.current_target_rho:.1f} '
            f'alpha={self.current_target_alpha:.1f} '
            f'z={self.current_target_z:.1f} '
            f'p2={joint_target.wrist:.1f} '
            f'p3={joint_target.elbow:.1f} '
            f'p4={joint_target.shoulder:.1f} '
            f'p5={joint_target.base:.1f}'
        )

    def command_observe_pose(self):
        self.current_target_rho = BASE_MIN_RHO
        self.current_target_alpha = 0.0
        self.current_target_z = START_PICKUP_Z
        self.detection_history.clear()
        self.out_of_reach_counter = 0
        self.alignment_error_history.clear()
        self.command_gripper(OPEN_GRIPPER_ANGLE)
        self.command_planar_target(
            rho=self.current_target_rho,
            alpha_deg=self.current_target_alpha,
            z=self.current_target_z,
            new_state=State.MOVING_TO_OBSERVE,
        )

    def command_idle_pose(self, new_state: State):
        idle_position = IDLE_POSE.copy()
        self.publish_arm_control(idle_position, new_state)


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
        error_magnitude = abs(error_x) + abs(error_y)

        aligned = abs(error_x) <= ALIGN_X_TOLERANCE and abs(error_y) <= ALIGN_Y_TOLERANCE

        if self.current_target_z > ALIGNMENT_Z:
            # Above ALIGNMENT_Z: align step-by-step
            if aligned:
                self.out_of_reach_counter = 0
                self.alignment_error_history.clear()
                new_z = max(FINAL_PICKUP_Z, self.current_target_z - DESCENT_STEP_MM)
                rho = self.current_target_rho
                alpha = self.current_target_alpha
            else:
                self.alignment_error_history.append(error_magnitude)
                new_z = self.current_target_z
                pixel_to_mm = PIXEL_TO_MM * (self.current_target_z / IDLE_Z)
                delta_rho = self.clamp_step(error_y * pixel_to_mm, MAX_RHO_STEP_MM)
                delta_alpha = self.clamp_step(error_x * PIXEL_TO_ALPHA_DEG, MAX_ALPHA_STEP_DEG)
                requested_rho = self.current_target_rho + delta_rho
                requested_alpha = self.current_target_alpha + delta_alpha

                if self.is_out_of_reach_adjustment(requested_rho, requested_alpha):
                    self.out_of_reach_counter += 1
                    if self.out_of_reach_counter >= OUT_OF_REACH_CONFIRMATION_STEPS:
                        self.track_only_mode = False
                        self.alignment_error_history.clear()
                        self.transition_to(State.IDLE)
                        self.publish_result('PICK_UP_FAIL_OUT_OF_REACH')
                        return
                else:
                    self.out_of_reach_counter = 0

                if self.is_alignment_stalled():
                    self.alignment_error_history.clear()
                    new_z = max(FINAL_PICKUP_Z, self.current_target_z - DESCENT_STEP_MM)
                    rho = self.current_target_rho
                    alpha = self.current_target_alpha
                else:
                    rho = requested_rho
                    alpha = requested_alpha
        else:
            # Below ALIGNMENT_Z: just descend to FINAL_PICKUP_Z without aligning
            self.out_of_reach_counter = 0
            self.alignment_error_history.clear()
            new_z = max(FINAL_PICKUP_Z, self.current_target_z - DESCENT_STEP_MM)
            rho = self.current_target_rho
            alpha = self.current_target_alpha
            if new_z == FINAL_PICKUP_Z:
                # At FINAL_PICKUP_Z, close gripper
                if self.track_only_mode:
                    self.track_only_mode = False
                    self.transition_to(State.IDLE)
                    self.publish_result(
                        f'TRACK_ONLY_SUCCESS x={detection.center_x} y={detection.center_y} '
                        f'rho={self.current_target_rho:.1f} alpha={self.current_target_alpha:.1f} '
                        f'z={self.current_target_z:.1f}'
                    )
                    return
                else:
                    self.command_gripper(CLOSED_GRIPPER_ANGLE, new_state=State.CLOSING_GRIPPER)
                    return

        try:
            self.command_planar_target(
                rho=rho,
                alpha_deg=alpha,
                z=new_z,
                new_state=State.ALIGNING,
            )
        except ValueError as exc:
            if self.current_target_z > FINAL_PICKUP_Z:
                # If joint limits reached at high Z, descend and try again at lower height
                fallback_z = max(FINAL_PICKUP_Z, self.current_target_z - DESCENT_STEP_MM)
                try:
                    self.command_planar_target(
                        rho=self.current_target_rho,
                        alpha_deg=self.current_target_alpha,
                        z=fallback_z,
                        new_state=State.ALIGNING,
                    )
                except ValueError:
                    # If still fails, then error
                    self.track_only_mode = False
                    self.transition_to(State.ERROR)
                    self.publish_result(f'PICK_UP_FAIL_RANGE: {exc}')
            else:
                self.track_only_mode = False
                self.transition_to(State.ERROR)
                self.publish_result(f'PICK_UP_FAIL_RANGE: {exc}')

    def publish_arm_control(self, target_position: list[float], new_state: State | None = None):
        self.new_position = [float(value) for value in target_position]

        msg = ArmControl()
        msg.position = self.new_position
        msg.time = self.compute_move_times(self.position, self.new_position)
        self.control_pub.publish(msg)

        self.position = self.new_position.copy()
        max_move_time_ms = max(msg.time) if len(msg.time) > 0 else MIN_TIME_MS
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

    def should_log_detection(self, detection: VisionDetection) -> bool:
        if self.last_logged_detection is None or self.last_vision_log_time is None:
            return True

        age = self.get_clock().now() - self.last_vision_log_time
        changed_enough = (
            abs(detection.center_x - self.last_logged_detection.center_x) >= VISION_LOG_DELTA_PIXELS
            or abs(detection.center_y - self.last_logged_detection.center_y) >= VISION_LOG_DELTA_PIXELS
        )
        return changed_enough and age >= Duration(seconds=VISION_LOG_MIN_INTERVAL_SEC)

    def clamp_step(self, value: float, limit: float) -> float:
        return max(-limit, min(limit, value))

    def is_out_of_reach_adjustment(self, requested_rho: float, requested_alpha: float) -> bool:
        requested_min_rho = get_min_rho(requested_alpha, requested_rho)
        return requested_rho > MAX_RHO or requested_rho < requested_min_rho

    def is_alignment_stalled(self) -> bool:
        if len(self.alignment_error_history) < self.alignment_error_history.maxlen:
            return False

        improvement = self.alignment_error_history[0] - self.alignment_error_history[-1]
        return improvement < OUT_OF_REACH_MIN_ERROR_IMPROVEMENT

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
