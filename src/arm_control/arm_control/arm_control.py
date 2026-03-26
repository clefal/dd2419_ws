import rclpy
import time
from rclpy.node import Node
from robp_interfaces.msg import ArmControl
from std_msgs.msg import String

#TODO: make it faster??

class Arm_control(Node):
    def __init__(self):
        super().__init__('arm_control')
        self.in_start_position = False
        self.holding_object = False
        self.pub = self.create_publisher(ArmControl, '/arm/control', 10)

        self.res = self.create_publisher(String, '/arm/result', 10)

        self.subscription = self.create_subscription(
            String,
            '/arm/action',       
            self.change_position,   
            10                    
        )

    def send_msg_start_position(self):
        msg = ArmControl()
        msg.position[0] = 10
        msg.position[1] = 60
        msg.position[2] = 170
        msg.position[3] = 220
        msg.position[4] = 180
        self.pub.publish(msg)

        time.sleep(3.0)

        msg = ArmControl()
        msg.position[0] = 10
        msg.position[1] = 90
        msg.position[2] = 170
        msg.position[3] = 210
        msg.position[4] = 120
        self.pub.publish(msg)

        time.sleep(3.0)

        msg = ArmControl()
        msg.position[0] = 10
        msg.position[2] = 50
        msg.position[3] = 210
        msg.position[4] = 120
        self.pub.publish(msg)
        time.sleep(3.0)

        self.in_start_position = True


    def send_msg_start_position_after_dropoff(self):

        msg = ArmControl()
        msg.position[0] = 10
        msg.position[2] = 150
        msg.position[3] = 210
        msg.position[4] = 120
        self.pub.publish(msg)

        time.sleep(3.0)

        msg = ArmControl()
        msg.position[0] = 10
        msg.position[2] = 50
        msg.position[3] = 210
        msg.position[4] = 120
        self.pub.publish(msg)
        time.sleep(3.0)

        self.in_start_position = True



    def send_msg_raise_camera(self):
        msg = ArmControl()
        msg.position[0] = 10
        msg.position[2] = 150
        msg.position[3] = 190
        msg.position[4] = 120

        self.pub.publish(msg)
        time.sleep(3.0)

    def send_msg_lower_camera(self):
        msg = ArmControl()
        msg.position[0] = 10
        msg.position[2] = 50
        msg.position[3] = 210
        msg.position[4] = 120

        self.pub.publish(msg)
        time.sleep(3.0)

    def send_msg_close_grip(self):
        msg = ArmControl()
        msg.position[0] = 100
        msg.position[2] = 150
        msg.position[3] = 190
        msg.position[4] = 40

        self.pub.publish(msg)
        time.sleep(3.0)

    def send_msg_open_grip(self):
        msg = ArmControl()
        msg.position[0] = 10
        msg.position[2] = 150
        msg.position[3] = 190
        msg.position[4] = 120

        self.pub.publish(msg)
        time.sleep(3.0)

    def send_msg_lower_arm(self):
        msg = ArmControl()
        msg.position[0] = 10
        msg.position[2] = 150
        msg.position[3] = 190
        msg.position[4] = 40

        self.pub.publish(msg)
        time.sleep(3.0)

    def send_msg_to_box(self):
        msg = ArmControl()
        msg.position[0] = 100
        msg.position[2] = 150
        msg.position[3] = 190
        msg.position[4] = 65

        self.pub.publish(msg)
        time.sleep(3.0)

    def send_msg_raise_arm(self):
        msg = ArmControl()
        msg.position[0] = 100
        msg.position[2] = 150
        msg.position[3] = 190
        msg.position[4] = 120

        self.pub.publish(msg)
        time.sleep(3.0)

    def publish_res(self, data):
        msg = String()
        msg.data = data
        self.res.publish(msg)

    def start_position(self):
        self.send_msg_start_position()
        self.publish_res("START_SUCCESS")

    def pick_up_object(self):
        if (self.in_start_position):
            self.send_msg_raise_camera()
            time.sleep(3.0)

            self.send_msg_lower_arm()
            time.sleep(3.0)

            self.send_msg_close_grip()
            time.sleep(3.0)

            self.send_msg_raise_arm()
            time.sleep(3.0)

            self.in_start_position = False
            self.holding_object = True #TODO: check that an object is actually in arm
            self.publish_res("PICK_UP_SUCCESS")
            if (not self.holding_object):
                self.publish_res("PICK_UP_FAIL_NO_OBJECT")
                self.get_logger().error(f'Failed to pick up object: Object not in arm')

        else:
            self.publish_res("PICK_UP_FAIL_NO_START")
            self.get_logger().error(f'Can not initilize pick up: Arm not in start position')

    def drop_object(self):
        if (self.holding_object):
            self.send_msg_to_box()
            time.sleep(3.0)
            self.send_msg_open_grip()
            self.holding_object = False
            self.publish_res("DROP_SUCCESS")
            self.send_msg_start_position_after_dropoff()
            
        else:
            self.publish_res("DROP_FAIL_NO_OBJECT")
            self.get_logger().error(f'Can not drop object: Not holdning an object')

    def change_position(self, msg):    
        if msg.data == "START":
            self.start_position()
        elif msg.data == "PICK_UP":
            self.pick_up_object()
        elif msg.data == "DROP":
            self.drop_object()

def main():
    rclpy.init()
    node = Arm_control()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
