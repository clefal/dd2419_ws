import rclpy
import time
import numpy as np
from rclpy.node import Node
from robp_interfaces.msg import ArmControl
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Int32MultiArray
import cv2

#TODO: make it faster??

class Arm_control(Node):
    def __init__(self):
        super().__init__('arm_control')
        self.in_idle_position = False
        self.holding_object = False
        self.position = [40, 120, 30, 220, 180, 120]
        self.time = np.full((6), 3000)

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

        msg_out = Int32MultiArray()
        msg_out.data = [cx, cy]
        self.center_pub.publish(msg_out)


    def send_msg_start_position(self):
        msg = ArmControl()
        self.position[3] = 166.4
        msg.position = self.position
        msg.time = self.time
        self.control.publish(msg)
        time.sleep(max(msg.time)/1000)

        self.position[2] = 16.6
        self.position[4] = 89.8
        msg.position = self.position
        msg.time = self.time
        self.control.publish(msg)
        time.sleep(max(msg.time)/1000)

        self.in_idle_position = True

    def is_box_in_pickup_range(self):
        pass 
        #something camera something 

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
    print("hi")
    rclpy.init()
    node = Arm_control()
    node.send_msg_start_position()
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