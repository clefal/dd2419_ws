import rclpy
import time
from rclpy.node import Node
from robp_interfaces.msg import ArmControl

class Arm_control(Node):
    def __init__(self):
        super().__init__('arm_control')
        self.pub = self.create_publisher(ArmControl, '/arm/control', 10)

    def send_msg_start_position(self):
        msg = ArmControl()
        msg.position[0] = 10
        msg.position[2] = 170
        msg.position[3] = 200
        msg.position[4] = 120
        self.pub.publish(msg)


    def send_msg_close_grip(self):
        msg = ArmControl()
        msg.position[0] = 90
        self.pub.publish(msg)

    def send_msg_open_grip(self):
        msg = ArmControl()
        msg.position[0] = 10
        self.pub.publish(msg)

    def send_msg_lower_arm(self):
        msg = ArmControl()
        msg.position[4] = 40
        self.pub.publish(msg)

    def send_msg_raise_arm(self):
        msg = ArmControl()
        msg.position[4] = 120
        self.pub.publish(msg)
            

def main():
    rclpy.init()
    node = Arm_control()
    node.send_msg_start_position()
    time.sleep(3.0)

    node.send_msg_lower_arm()
    time.sleep(3.0)

    node.send_msg_close_grip()
    time.sleep(3.0)

    node.send_msg_raise_arm()
    time.sleep(3.0)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
