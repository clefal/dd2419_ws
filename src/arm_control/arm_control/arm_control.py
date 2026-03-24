import rclpy
import time
import math
import numpy as np
from rclpy.node import Node
from robp_interfaces.msg import ArmControl
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Int32MultiArray
import cv2


#TODO: have limits that work with rotational movement 
MAX_RHO = 190 #could prob be larger, like 194
MIN_RHO = 160 #could prob be smaller, like 154
Z = 15

#Arm part lengths 
L1 = 101
L2 = 94

#Arm time constants  
MS_PER_DEGREE = 60 #TODO adjust: could prob be smaller (aka faster/smoother)
MIN_TIME = 200 #TODO smaller?? 
MAX_TIME = 3000

#define idle position
IDLE_P2 = 16.6
IDLE_P3 = 166.4
IDLE_P4 = 89.8

#define holding position
HOLDING_P2 = 30
HOLDING_P3 = 170
HOLDING_P4 = 120

#more standard positions 
START_POSITION = [40, 120, 30, 220, 180, 120]
CLOSED_GRIPPER_ANGLE = 100
OPEN_GRIPPER_ANGLE = 40

#Cube constants 
MIN_CUBE_Y = 408
MAX_CUBE_Y = 438
PIXEL_TO_MM = 0.217

#Detection constants 
REQUIRED_DETECTIONS = 3
DETECTION_TOLERANCE = 1

class Arm_control(Node):
    def __init__(self):
        super().__init__('arm_control')
        self.state = "start"
        self.pickup_ready = False
        self.in_idle_position = False
        self.holding_object = False
        self.position = START_POSITION.copy()
        self.new_position = START_POSITION.copy()
        self.time = np.full((6), 3000)
        self.rho = 0
        self.cube_y = 0
        self.last_detections = []

        self.bridge = CvBridge()
        self.center_pub = self.create_publisher(Int32MultiArray, '/green_cube_center', 10)

        #To visialize the cube detection 
        self.mask_pub = self.create_publisher(Image, '/green_mask', 10)

        self.control = self.create_publisher(ArmControl, '/arm/control', 10)

        self.image_subscription = self.create_subscription(
            Image,
            '/arm/camera/image_raw',
            self.image_callback,
            10
        )

        self.res = self.create_publisher(String, '/arm/result', 10)

        self.action_subscription = self.create_subscription(
            String,
            '/arm/action',       
            self.action,   
            10                    
        )

    def image_callback(self, msg: Image):
        if self.state not in ["detect"]:
            return

        if msg.encoding == 'bgr8':
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        elif msg.encoding == 'yuv422_yuy2':
            yuy = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 2))
            frame = cv2.cvtColor(yuy, cv2.COLOR_YUV2BGR_YUY2)
        else:
            raise NotImplementedError(f"Encoding {msg.encoding} not supported")

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
        self.mask_pub.publish(mask_msg)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return  

        largest_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest_contour) < 500:
            return

        M = cv2.moments(largest_contour)
        if M['m00'] == 0:
            return

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        self.get_logger().info(f"Green cube center at: x={cx}, y={cy}")

        #TODO add error handling so arm does not get stuck in pick up 
        # ( should use funciton self.send_msg_failed_pickup_to_idle() )
        # Maybe just a timeout is fine? 

        self.last_detections.append(cy)

        if len(self.last_detections) > REQUIRED_DETECTIONS:
            self.last_detections.pop(0)

        if len(self.last_detections) == REQUIRED_DETECTIONS:
            #if detection is stable 
            if max(self.last_detections) - min(self.last_detections) < DETECTION_TOLERANCE:
                stable_y = int(sum(self.last_detections) / REQUIRED_DETECTIONS)
                self.cube_y = stable_y

                # Cube is within range to pick up 
                if self.cube_y > MIN_CUBE_Y and self.cube_y < MAX_CUBE_Y:
                    self.state = "pickup"
                    self.pick_up_object()
                else:
                # Adjust arm to center cube 
                    self.send_msg_adjust_pickup()
                    self.state = "detect"

                self.last_detections.clear()

        msg_out = Int32MultiArray()
        msg_out.data = [cx, cy]
        self.center_pub.publish(msg_out)

    """
    solves equation to give arm angles for a given rho and z 
    (limited to orientation looking straight down)
    input: rho, z
    output: position[4], position[3], position[2]
    """
    def calc_arm_angles(self, rho, z):
        orientation = math.radians(-90)
        if rho > MAX_RHO or rho < MIN_RHO:
            #TODO: correctly raise error 
            print("bad rho")
            return "error"

        r2 = rho*rho + z*z
        cos_t2 = (r2 - L1**2 - L2**2) / (2 * L1 * L2)

        if abs(cos_t2) > 1:
            #TODO: correctly raise error 
            print("position unreachable")
            return "error"
        
        t2 = -math.acos(cos_t2)

        k1 = L1 + L2 * math.cos(t2)
        k2 = L2 * math.sin(t2)

        t1 = math.atan2(z, rho) - math.atan2(k2, k1)

        t3 = orientation - t1 - t2

        #Convert solution angles to robot arm angles 
        s1 = 30 + math.degrees(t1)
        s2 = 120 - math.degrees(t2)
        s3 = 120 + math.degrees(t3)

        if s1 < 30 or s1 > 120 or s2 < 100 or s2 > 210 or s3 < 17 or s3 > 140:
            #TODO: correctly raise error 
            print("bad angles:", s1, s2, s3)
            return "error"
        
        print(s1, s2, s3)
        return s1, s2, s3 

    """Update self.time to match angle delta"""
    def set_time(self):
        self.time = []

        for i in range(len(self.position)):
            diff = abs(self.new_position[i] - self.position[i])
            t = diff * MS_PER_DEGREE
            t = max(MIN_TIME, min(MAX_TIME, int(t)))
            self.time.append(t)
    
    """Publish arm control message with currrent self.new_position and self.time 
    Sleeps for the duration of longest arm movement to avoid concurrent arm messages 
    """
    def publish_arm_control(self):
        msg = ArmControl()
        msg.position = self.new_position
        self.set_time()
        print(self.time)
        msg.time = self.time 
        self.control.publish(msg)
        time.sleep(max(msg.time)/1000)
        self.position = self.new_position.copy()

    """Publish result message to /arm/result"""
    def publish_res(self, data):
        msg = String()
        msg.data = data
        self.res.publish(msg)

    """return true if object is in arm else return false"""
    def is_holding_object(self):
        #TODO prob use a like pickup_check state or something in the camera callback but idk
        return True
    
    """
    use arm camera in idle position to determine if there (1) is a cube and (2) if cube is within range 
    if not (1) return no cube detection error 
    if not (2) return cube outside of range error 
    else return true
    """
    def is_box_in_pickup_range(self):
        #TODO
        #May also need a new state but idk
        return True

    """Robot goes into idle position from start position (only run at inilization)"""
    def send_msg_initalize_position(self):
        self.new_position[3] = IDLE_P3
        self.publish_arm_control()

        self.new_position[2] = IDLE_P2
        self.new_position[4] = IDLE_P4
        self.publish_arm_control()

    """Moves from idle_postion to inital_pickup position to detect obejct"""
    def send_msg_idle_to_detect(self):
        middle_rho = (MIN_RHO + MAX_RHO) / 2
        p4, p3, p2 = self.calc_arm_angles(middle_rho, Z)
        self.new_position[2:5] = [p2, p3, p4]
        self.publish_arm_control()

        self.rho = middle_rho

    """A return to idle postion from pickup position (used when detection times out)"""
    def send_msg_detect_to_idle(self):
        #Go to middle pickup
        middle_rho = (MIN_RHO + MAX_RHO) / 2
        p4, p3, p2 = self.calc_arm_angles(middle_rho, Z)
        self.new_position[2:5] = [p2, p3, p4]
        self.publish_arm_control()

        #Go to idle 
        self.new_position[2] = IDLE_P2
        self.new_position[3] = IDLE_P3
        self.new_position[4] = IDLE_P4
        self.publish_arm_control()

    def send_msg_pickup_to_holding(self):
        self.new_position[4] = HOLDING_P4
        self.publish_arm_control()

        self.new_position[2] = HOLDING_P2
        self.new_position[3] = HOLDING_P3
        self.publish_arm_control()

    """When arm in holding position without holding cube, open gripper and return to Idle"""
    def send_msg_holding_to_idle(self):
        self.new_position[0] = OPEN_GRIPPER_ANGLE
        self.new_position[2] = IDLE_P2
        self.new_position[3] = IDLE_P3
        self.new_position[4] = IDLE_P4
        self.publish_arm_control()

    """Move from idle position to drop of position"""
    def send_msg_holding_to_drop(self):
        pass
        #TODO
        #Go into a drop off position 
        #later TODO include box detection?? 

    """Move from drop position to idle position"""
    def send_msg_drop_to_idle(self):
        pass
        #TODO
        #Change postion to idle 
        
    """Adjust pick-up position based on camera feedback"""
    def send_msg_adjust_pickup(self):
        cube_middle = (MIN_CUBE_Y + MAX_CUBE_Y) / 2
        cube_diff =  cube_middle - self.cube_y
        new_rho = self.rho + cube_diff * PIXEL_TO_MM

        p4, p3, p2 = self.calc_arm_angles(new_rho, Z)
        self.new_position[2:5] = [p2, p3, p4]
        self.publish_arm_control()
        self.rho = new_rho

    def send_msg_close_gripper(self):
        self.new_position[0] = CLOSED_GRIPPER_ANGLE
        self.publish_arm_control()

    def send_msg_open_gripper(self):
        self.new_position[0] = OPEN_GRIPPER_ANGLE
        self.publish_arm_control()

    def pick_up_object(self):
        self.send_msg_close_gripper()
        self.send_msg_pickup_to_holding()
        if self.is_holding_object():
            self.state = "holding"
            self.publish_res("PICK_UP_SUCCESS")
        else:
            self.publish_res("PICK_UP_FAIL_NO_OBJECT")
            self.send_msg_holding_to_idle()
            self.state = "idle"
            #TODO handle error ???

    def initialize(self):
        if self.state == "start":
            self.send_msg_initalize_position()
            self.publish_res("START_SUCCESS")
            self.state = "idle"
        else:
            self.publish_res("START_FAIL")

    def initialize_pickup_process(self):
        if self.state == "idle":
            if self.is_box_in_pickup_range():
                self.send_msg_idle_to_detect()
                self.state = "detect"
            else:
                self.publish_res("PICK_UP_FAIL_RANGE") #TODO maybe different error for cube not in range or can't see cube at all
        else: 
            self.publish_res("PICK_UP_FAIL_NO_IDLE")

    def drop_object(self):
        if self.state == "holding":
            self.state = "drop"
            self.send_msg_holding_to_drop()
            self.send_msg_open_gripper()
            self.send_msg_drop_to_idle()
            self.state = "idle"
            self.publish_res("DROP_SUCCESS")
        else: 
            self.publish_res("DROP_FAIL_NO_OBJECT")

    """Perform actions based on goal manager"""
    def action(self, msg):
        if msg.data == "START":
            self.initialize()
        elif msg.data == "PICK_UP":
            self.initialize_pickup_process()
        elif msg.data == "DROP":
            self.drop_object()

def main():
    rclpy.init()
    node = Arm_control()
    test_pick_up_min_range(node)
    #test_pick_up_max_range(node)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

def test_pick_up_min_range(node):
    node.send_msg_initalize_position()
    node.send_msg_idle_to_detect()
    p4, p3, p2 = node.calc_arm_angles(MIN_RHO, Z)
    node.new_position[2:5] = [p2, p3, p4]
    node.publish_arm_control()
    node.pick_up_object()
    
def test_pick_up_max_range(node):
    node.send_msg_initalize_position()
    node.send_msg_idle_to_detect()
    p4, p3, p2 = node.calc_arm_angles(MAX_RHO, Z)
    node.new_position[2:5] = [p2, p3, p4]
    node.publish_arm_control()
    node.pick_up_object()

if __name__ == '__main__':
    main()