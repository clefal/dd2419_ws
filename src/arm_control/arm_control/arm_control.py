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

#BIG TODO LIST 
# Correct error handling of get_arm_angles 

# is_box_in_pickup_range()


#TODO: have limits that work with rotational movement 
MAX_RHO = 190 #could prob be larger, like 194
MIN_RHO = 160 #could prob be smaller, like 154
Z = 15

#Arm part lengths 
L1 = 101
L2 = 94

#Arm time constraints 
MS_PER_DEGREE = 60
MIN_TIME = 200
MAX_TIME = 3000

#define idle position
IDLE_P2 = 16.6
IDLE_P3 = 166.4
IDLE_P4 = 89.8

MIN_CUBE_Y = 408
MAX_CUBE_Y = 438
PIXEL_TO_MM = 0.217

class Arm_control(Node):
    def __init__(self):
        super().__init__('arm_control')
        self.in_idle_position = False
        self.holding_object = False
        self.position = [40, 120, 30, 220, 180, 120]
        self.new_position = [40, 120, 30, 220, 180, 120]
        self.time = np.full((6), 3000)
        self.rho = 0
        self.cube_y = 0

        self.bridge = CvBridge()
        self.center_pub = self.create_publisher(Int32MultiArray, '/green_cube_center', 10)

        self.mask_pub = self.create_publisher(Image, '/green_mask', 10)

        self.control = self.create_publisher(ArmControl, '/arm/control', 10)

        self.subscription = self.create_subscription(
            Image,
            '/arm/camera/image_raw',
            self.image_callback,
            10
        )

        # self.res = self.create_publisher(String, '/arm/result', 10)

        # self.subscription = self.create_subscription(
        #     String,
        #     '/arm/action',       
        #     self.change_position,   
        #     10                    
        # )

    def image_callback(self, msg: Image):
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
        self.cube_y = cy

        msg_out = Int32MultiArray()
        msg_out.data = [cx, cy]
        self.center_pub.publish(msg_out)

    """
    Publish arm control message with currrent self.new_position and self.time 
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

    """Update self.time to match angle delta"""
    def set_time(self):
        self.time = []

        for i in range(len(self.position)):
            diff = abs(self.new_position[i] - self.position[i])
            t = diff * MS_PER_DEGREE
            t = max(MIN_TIME, min(MAX_TIME, int(t)))
            self.time.append(t)

    """Robot goes into idle position from start position (only run at inilization)"""
    def send_msg_initalize_position(self):
        self.new_position[3] = IDLE_P3
        self.publish_arm_control()

        self.new_position[2] = IDLE_P2
        self.new_position[4] = IDLE_P4
        self.publish_arm_control()

        self.in_idle_position = True

    """Moves from idle_postion to inital_pickup position"""
    def send_msg_idle_to_pickup(self):
        middle_rho = (MIN_RHO + MAX_RHO) / 2
        p4, p3, p2 = self.calc_arm_angles(middle_rho, Z)
        self.new_position[2:5] = [p2, p3, p4]
        self.publish_arm_control()

        self.in_idle_position = False
        self.rho = middle_rho

    """Adjust pick-up position based on camera feedback"""
    def send_msg_adjust_pickup(self):
        cube_middle = (MIN_CUBE_Y + MAX_CUBE_Y) / 2
        cube_diff =  cube_middle - self.cube_y
        new_rho = self.rho + cube_diff * PIXEL_TO_MM

        p4, p3, p2 = self.calc_arm_angles(new_rho, Z)
        self.new_position[2:5] = [p2, p3, p4]
        #self.publish_arm_control()
        #self.rho = new_rho    

        print(new_rho) 
            


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

        #TODO: error check that s1-3 are okey values
        if s1 < 30 and s1 > 120 and s2 < 100 and s2 > 210 and s3 < 17  and s3 > 140:
            #TODO: correctly raise error 
            print("bad angles:", s1, s2, s3)
            return "error"
        
        print(s1, s2, s3)
        return s1, s2, s3        

    def is_box_in_pickup_range(self):
        pass 
        #TODO something camera something 

    #TODO: def send_msg_idle_position(self):

    # def publish_res(self, data):
    #     msg = String()
    #     msg.data = data
    #     self.res.publish(msg)

    # def start_position(self):
    #     self.send_msg_start_position()
    #     self.publish_res("START_SUCCESS")

    # def pick_up_object(self):
    #     if (self.in_start_position):
    #         self.send_msg_raise_camera()
    #         time.sleep(3.0)

    #         self.send_msg_lower_arm()
    #         time.sleep(3.0)

    #         self.send_msg_close_grip()
    #         time.sleep(3.0)

    #         self.send_msg_raise_arm()
    #         time.sleep(3.0)

    #         self.in_start_position = False
    #         self.holding_object = True #TODO: check that an object is actually in arm
    #         self.publish_res("PICK_UP_SUCCESS")
    #         if (not self.holding_object):
    #             self.publish_res("PICK_UP_FAIL_NO_OBJECT")
    #             self.get_logger().error(f'Failed to pick up object: Object not in arm')

    #     else:
    #         self.publish_res("PICK_UP_FAIL_NO_START")
    #         self.get_logger().error(f'Can not initilize pick up: Arm not in start position')

    # def drop_object(self):
    #     if (self.holding_object):
    #         self.send_msg_open_grip()
    #         self.holding_object = False
    #         self.send_msg_start_position()
    #         self.publish_res("DROP_SUCCESS")
    #     else:
    #         self.publish_res("DROP_FAIL_NO_OBJECT")
    #         self.get_logger().error(f'Can not drop object: Not holdning an object')

    # def change_position(self, msg):    
    #     if msg.data == "START":
    #         self.start_position()
    #     elif msg.data == "PICK_UP":
    #         self.pick_up_object()
    #     elif msg.data == "DROP":
    #         self.drop_object()

def main():
    rclpy.init()
    node = Arm_control()
    node.send_msg_initalize_position()
    node.send_msg_idle_to_pickup()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()

# def send_msg_raise_camera(self):
#     msg = ArmControl()
#     msg.position[0] = 10
#     msg.position[2] = 150
#     msg.position[3] = 190
#     msg.position[4] = 120

#     self.pub.publish(msg)
#     time.sleep(3.0)

# def send_msg_lower_camera(self):
#     msg = ArmControl()
#     msg.position[0] = 10
#     msg.position[2] = 50
#     msg.position[3] = 210
#     msg.position[4] = 120

#     self.pub.publish(msg)
#     time.sleep(3.0)

# def send_msg_close_grip(self):
#     msg = ArmControl()
#     msg.position[0] = 100
#     msg.position[2] = 150
#     msg.position[3] = 190
#     msg.position[4] = 40

#     self.pub.publish(msg)
#     time.sleep(3.0)

# def send_msg_open_grip(self):
#     msg = ArmControl()
#     msg.position[0] = 10
#     msg.position[2] = 150
#     msg.position[3] = 190
#     msg.position[4] = 120

#     self.pub.publish(msg)
#     time.sleep(3.0)

# def send_msg_lower_arm(self):
#     msg = ArmControl()
#     msg.position[0] = 10
#     msg.position[2] = 150
#     msg.position[3] = 190
#     msg.position[4] = 40

#     self.pub.publish(msg)
#     time.sleep(3.0)

# def send_msg_raise_arm(self):
#     msg = ArmControl()
#     msg.position[0] = 100
#     msg.position[2] = 150
#     msg.position[3] = 190
#     msg.position[4] = 120

#     self.pub.publish(msg)
#     time.sleep(3.0)