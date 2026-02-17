import rclpy
import time
from rclpy.node import Node
from robp_interfaces.msg import ArmControl

#TODO: make it faster??

class Arm_control(Node):
    def __init__(self):
        super().__init__('arm_control')
        self.pub = self.create_publisher(ArmControl, '/arm/control', 10)
        self.in_start_position = False
        self.holding_object = False

    def send_msg_start_position(self):
        msg = ArmControl()
        msg.position[0] = 40
        msg.position[2] = 170
        msg.position[3] = 220
        msg.position[4] = 180
        self.pub.publish(msg)

        time.sleep(3.0)

        msg = ArmControl()
        msg.position[0] = 10
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

    def send_msg_raise_arm(self):
        msg = ArmControl()
        msg.position[0] = 100
        msg.position[2] = 150
        msg.position[3] = 190
        msg.position[4] = 120

        self.pub.publish(msg)
        time.sleep(3.0)

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
            self.holding_object = True
            #TODO: check that an object is in arm, and change self.holding_obejct property
            if (not self.holding_object):
                #TODO: handle error 
                self.get_logger().error(f'Failed to pick up object: Object not in arm')

        else:
            #TODO: handle error (Like tell task manager there was an error and to try again???)
            self.get_logger().error(f'Can not initilize pick up: Arm not in start position')
            self.send_msg_start_position()

    def drop_object(self):
        if (self.holding_object):
            self.send_msg_open_grip()
            self.holding_object = False
            self.send_msg_start_position()
        else:
            #TODO: handle error
            self.get_logger().error(f'Can not drop object: Not holdning an object')

def main():
    rclpy.init()
    node = Arm_control()

    #TODO: Listen to status topic 
    
    #TODO: When we get start message call node.send_msg_start_position()
    #TODO: When we get arrived message call node.pick_up_object() 
    #TODO: When we get drop message call node.drop_object()
   
    rclpy.shutdown()

if __name__ == '__main__':
    main()
