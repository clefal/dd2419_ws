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
    OPEN_GRIPPER_ANGLE,
    CLOSED_GRIPPER_ANGLE,
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
    make_planar_target,
    planar_to_joint_target,
)

from arm_control.arm_vision import (
    HOLDING_CHECK_TOPIC,
    HOLDING_ANSWER_TOPIC,
    CHECK_HOLDING_MSG, 
    HOLDING_SUCCESS_MSG,
    HOLDING_FAIL_MSG
) 

#Time for joint movements 
MS_PER_DEGREE = 45
MIN_TIME_MS = 150
MAX_TIME_MS = 2000

#TOPICS 
VISION_TOPIC = '/arm/vision/cube_center'
ACTION_TOPIC = '/arm/action'
RESULT_TOPIC = '/arm/result'
CONTROL_TOPIC = '/arm/control'

CONTROL_RATE_HZ = 10.0

TARGET_PIXEL_X = 310
TARGET_PIXEL_Y = 420 #400
ALIGN_X_TOLERANCE = 15  #25
ALIGN_Y_TOLERANCE = 10
PIXEL_TO_MM = 0.22   #0.15
PIXEL_TO_ALPHA_DEG = 0.055  #0.055
MAX_RHO_STEP_MM = 6.0
MAX_ALPHA_STEP_DEG = 1.0    #2.0

DESCENT_STEP_MM = 5.0
FINAL_PICKUP_Z = DEFAULT_PICKUP_Z   #  current low value
START_PICKUP_Z = IDLE_Z - 50.0  # higher starting point
ALIGNMENT_Z = FINAL_PICKUP_Z + 10.0  # stop aligning below this Z to avoid vision issues

#STABLE DETECTION PARAMETERS
REQUIRED_DETECTIONS = 4 #3
STABLE_X_TOLERANCE = 5
STABLE_Y_TOLERANCE = 5

VISION_TIMEOUT_SEC = 2.0 #1 


class State(Enum):
    START = 'START'
    MOVING_TO_START_SAFE = 'MOVING_TO_START_SAFE'
    RETURN_TO_IDLE = 'RETURN_TO_IDLE'
    MOVING_TO_IDLE = 'MOVING_TO_IDLE'
    IDLE = 'IDLE'
    MOVING_TO_OBSERVE = 'MOVING_TO_OBSERVE'
    ALIGNING = 'ALIGNING'
    CLOSING_GRIPPER = 'CLOSING_GRIPPER'
    LIFTING = 'LIFTING'
    HOLDING = 'HOLDING'
    MOVING_TO_DROP = 'MOVING_TO_DROP'
    OPENING_FOR_DROP = 'OPENING_FOR_DROP'
    PICK_UP_TO_IDLE = 'PICK_UP_TO_IDLE'
    WAITING_FOR_HOLD_CONFIRM = 'WAITING_FOR_HOLD_CONFIRM'
    CHECK_HOLDING = 'CHECK_HOLDING'

class Result(Enum):
    IDLE_SUCCESS = 'IDLE_SUCCESS'
    DROP_SUCCESS = 'DROP_SUCCESS'
    PICK_UP_SUCCESS = 'PICK_UP_SUCCESS'
    START_FAIL = 'START_FAIL'
    PICK_UP_FAIL_NO_IDLE = 'PICK_UP_FAIL_NO_IDLE'
    PICK_UP_FAIL_NO_HOLDING = 'PICK_UP_FAIL_NO_HOLDING'
    PICK_UP_FAIL_NO_DETECTION = 'PICK_UP_FAIL_NO_DETECTION'
    PICK_UP_FAIL_OUT_OF_REACH = 'PICK_UP_FAIL_OUT_OF_REACH'
    DROP_FAIL_NO_OBJECT = 'DROP_FAIL_NO_OBJECT'

@dataclass
class VisionDetection:
    center_x: int
    center_y: int
    angle: int # rotation in degrees where negative is left and positive right

class ArmControlNode(Node):
    def __init__(self):
        super().__init__('arm_control')

        self.state = State.START
        self.position = [float(value) for value in INITIAL_POSITION]
        self.new_position = self.position.copy()
        self.motion_complete_time = self.get_clock().now()

        self.current_target_rho = None
        self.current_target_alpha = None
        self.current_target_z = None

        #Detection tracking
        self.latest_detection = None
        self.latest_detection_time = None
        self.detection_history = deque(maxlen=REQUIRED_DETECTIONS)

        #Publishers
        self.control_pub = self.create_publisher(ArmControl, CONTROL_TOPIC, 10)
        self.result_pub = self.create_publisher(String, RESULT_TOPIC, 10)
        self.holding_pub = self.create_publisher(String, HOLDING_CHECK_TOPIC, 10)

        #Subscriptions
        self.create_subscription(String, HOLDING_ANSWER_TOPIC, self.holding_answer_callback, 10)
        self.create_subscription(String, ACTION_TOPIC, self.action_callback, 10)
        self.create_subscription(Int32MultiArray, VISION_TOPIC, self.vision_callback, 10)

        #Control timer
        self.control_timer = self.create_timer(1.0 / CONTROL_RATE_HZ, self.control_loop)

    def action_callback(self, msg: String):
        command = msg.data.strip().upper()
        if command == 'START':
            self.handle_start_command()
        elif command == 'PICK_UP':
            self.handle_pickup_command()
        elif command == 'DROP':
            self.handle_drop_command()

    def control_loop(self):
        if self.is_motion_active():
            return
        
        if self.state == State.MOVING_TO_START_SAFE or self.state == State.RETURN_TO_IDLE:
            self.command_named_pose(IDLE_POSE)
            self.transition_to(State.MOVING_TO_IDLE)
            return

        if self.state == State.MOVING_TO_IDLE:
            self.transition_to(State.IDLE)
            self.publish_result(Result.IDLE_SUCCESS)
            return

        if self.state == State.MOVING_TO_OBSERVE:
            self.transition_to(State.ALIGNING)
            return

        if self.state == State.ALIGNING:
            self.update_alignment()
            return
        
        if self.state == State.CLOSING_GRIPPER:
            self.command_named_pose(LIFTING_POSE)
            self.transition_to(State.LIFTING)
            return

        if self.state == State.LIFTING:
            self.command_named_pose(HOLDING_POSE)
            self.transition_to(State.CHECK_HOLDING)
            return
        
        if self.state == State.CHECK_HOLDING:
            msg = String()
            msg.data = CHECK_HOLDING_MSG
            self.result_pub.publish(msg)
            self.transition_to(State.WAITING_FOR_HOLD_CONFIRM)
            return

        if self.state == State.MOVING_TO_DROP:
            self.command_gripper(OPEN_GRIPPER_ANGLE)
            self.transition_to(State.OPENING_FOR_DROP)
            return

        if self.state == State.OPENING_FOR_DROP:
            self.command_named_pose(IDLE_POSE)
            self.transition_to(State.MOVING_TO_IDLE)
            self.publish_result(Result.DROP_SUCCESS)
            return

    def handle_start_command(self):
        if self.state != State.START or self.is_motion_active():
            self.publish_result(Result.START_FAIL)
            return

        self.command_named_pose(START_SAFE_POSITION)
        self.transition_to(State.MOVING_TO_START_SAFE)

    def handle_pickup_command(self):
        if self.state != State.IDLE:
            self.publish_result(Result.PICK_UP_FAIL_NO_IDLE)
            return

        self.command_observe_pose()
        self.transition_to(State.MOVING_TO_OBSERVE)

    def handle_drop_command(self):
        if self.state != State.HOLDING:
            self.publish_result(Result.DROP_FAIL_NO_OBJECT)
            return

        self.command_named_pose(DROP_POSE)
        self.transition_to(State.MOVING_TO_DROP)

    #TODO: rotation
    def vision_callback(self, msg: Int32MultiArray):
        if len(msg.data) < 2:
            return

        detection = VisionDetection(center_x=int(msg.data[0]), center_y=int(msg.data[1]), angle=int(msg.data[2]))
        self.latest_detection = detection
        self.latest_detection_time = self.get_clock().now()
        self.detection_history.append(detection)

    def holding_answer_callback(self, msg):
        if self.state != State.WAITING_FOR_HOLD_CONFIRM:
            return
        if msg.data == HOLDING_SUCCESS_MSG:
            self.publish_result(Result.PICK_UP_SUCCESS)
            self.transition_to(State.HOLDING)
        elif msg.data == HOLDING_FAIL_MSG:
            self.publish_result(Result.PICK_UP_FAIL_NO_HOLDING)
            self.transition_to(State.RETURN_TO_IDLE)

    def command_observe_pose(self):
        self.current_target_rho = BASE_MIN_RHO
        self.current_target_alpha = 0.0
        self.current_target_z = START_PICKUP_Z
        self.detection_history.clear()
        self.command_gripper(OPEN_GRIPPER_ANGLE)
        self.command_planar_target(
            rho=self.current_target_rho,
            alpha_deg=self.current_target_alpha,
            z=self.current_target_z
        )

    def command_named_pose(self, pose):
        target_position = self.position.copy()

        for key in ['p0', 'p1', 'p2', 'p3', 'p4', 'p5']:
            if key in pose:
                idx = int(key[1])  # 'p2' -> 2
                target_position[idx] = pose[key]

        self.publish_arm_control(target_position)

    def command_gripper(self, angle):
        target_position = self.position.copy()
        target_position[0] = angle
        self.publish_arm_control(target_position)

    def command_planar_target(self, rho: float, alpha_deg: float, z: float):
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
        self.publish_arm_control(target_position)

    def update_alignment(self):
        detection = self.get_stable_detection()
        if detection is None:
            if self.vision_is_stale():
                self.transition_to(State.RETURN_TO_IDLE)
                self.publish_result(Result.PICK_UP_FAIL_NO_DETECTION)
            return

        error_x = TARGET_PIXEL_X - detection.center_x
        error_y = TARGET_PIXEL_Y - detection.center_y
        error_magnitude = abs(error_x) + abs(error_y)

        aligned = abs(error_x) <= ALIGN_X_TOLERANCE and abs(error_y) <= ALIGN_Y_TOLERANCE

        if self.current_target_z <= FINAL_PICKUP_Z:
            self.command_gripper(CLOSED_GRIPPER_ANGLE)
            self.transition_to(State.CLOSING_GRIPPER)
            return
        
        elif self.current_target_z <= ALIGNMENT_Z:
            if not aligned:
                self.transition_to(State.RETURN_TO_IDLE)
                self.publish_result(Result.PICK_UP_FAIL_OUT_OF_REACH)
                return
            else:
                new_z = FINAL_PICKUP_Z
                rho = self.current_target_rho
                alpha = self.current_target_alpha
        
        else:
            # Above ALIGNMENT_Z: align step-by-step
            if aligned:
                new_z = max(FINAL_PICKUP_Z, self.current_target_z - DESCENT_STEP_MM)
                rho = self.current_target_rho
                alpha = self.current_target_alpha
            else:
                new_z = self.current_target_z
                scale = max(0.4, self.current_target_z/ IDLE_Z)
                pixel_to_mm = PIXEL_TO_MM * scale
                delta_rho = self.clamp_step(error_y * pixel_to_mm, MAX_RHO_STEP_MM)
                delta_alpha = self.clamp_step(error_x * PIXEL_TO_ALPHA_DEG, MAX_ALPHA_STEP_DEG)
                rho = self.current_target_rho + delta_rho
                alpha = self.current_target_alpha + delta_alpha

        try:
            self.command_planar_target(
                rho=rho,
                alpha_deg=alpha,
                z=new_z
            )
        except ValueError as exc:
            if self.current_target_z > FINAL_PICKUP_Z:
                # If joint limits reached at high Z, descend and try again at lower height
                fallback_z = max(FINAL_PICKUP_Z, self.current_target_z - DESCENT_STEP_MM)
                try:
                    self.command_planar_target(
                        rho=self.current_target_rho,
                        alpha_deg=self.current_target_alpha,
                        z=fallback_z
                    )
                except ValueError:
                    self.transition_to(State.RETURN_TO_IDLE)
                    self.publish_result(Result.PICK_UP_FAIL_OUT_OF_REACH)
            else:
                self.transition_to(State.RETURN_TO_IDLE)
                self.publish_result(Result.PICK_UP_FAIL_OUT_OF_REACH)

    def publish_arm_control(self, target_position: list[float]):
        self.new_position = [float(value) for value in target_position]

        msg = ArmControl()
        msg.position = self.new_position
        msg.time = self.compute_move_times(self.position, self.new_position)
        self.control_pub.publish(msg)

        self.position = self.new_position.copy()
        max_move_time_ms = max(msg.time) if len(msg.time) > 0 else MIN_TIME_MS
        self.motion_complete_time = self.get_clock().now() + Duration(seconds=max_move_time_ms / 1000.0)

    def compute_move_times(self, current_position: list[float], target_position: list[float]) -> list[int]:
        move_times = []
        for current, target in zip(current_position, target_position):
            diff = abs(target - current)
            move_time = int(diff * MS_PER_DEGREE)
            move_time = max(MIN_TIME_MS, min(MAX_TIME_MS, move_time))
            move_times.append(move_time)
        return move_times

    #TODO: add rotation 
    def get_stable_detection(self):
        if len(self.detection_history) < REQUIRED_DETECTIONS:
            return None

        xs = [d.center_x for d in self.detection_history]
        ys = [d.center_y for d in self.detection_history]
        angles = [d.angle for d in self.detection_history]

        if max(xs) - min(xs) > STABLE_X_TOLERANCE:
            return None
        if max(ys) - min(ys) > STABLE_Y_TOLERANCE:
            return None


        center_x = sum(xs) // len(xs)
        center_y = sum(ys) // len(ys)
        angle = sum(angles) // len(angles)

        self.get_logger().info(
            f'Stable detection: x={center_x} y={center_y} angle={angle}'
        )

        return VisionDetection(
            center_x=center_x,
            center_y=center_y,
            angle=angle
        )

    def vision_is_stale(self) -> bool:
        if self.latest_detection_time is None:
            return True
        age = self.get_clock().now() - self.latest_detection_time
        return age > Duration(seconds=VISION_TIMEOUT_SEC)

    def is_motion_active(self):
        return self.get_clock().now() < self.motion_complete_time

    def clamp_step(self, value: float, limit: float) -> float:
        return max(-limit, min(limit, value))

    def transition_to(self, new_state: State):
        self.state = new_state
        self.get_logger().info(f'Arm state -> {new_state.value}')

    def publish_result(self, text: str):
        msg = String()
        msg.data = text.value
        self.result_pub.publish(msg)
        self.get_logger().info(f'Arm result: {text.value}')

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
