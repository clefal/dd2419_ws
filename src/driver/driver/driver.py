from pynput import keyboard
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from robp_interfaces.msg import DutyCycles

DUTY_CYCLE_MAX = 0.2
DUTY_CYCLE_MIN = -0.2

class Driver(Node):
    def __init__(self):
        super().__init__('driver')
        self.pub = self.create_publisher(DutyCycles, '/phidgets/motor/duty_cycles', 10)
        self.left_duty_cycle = 0.0
        self.right_duty_cycle = 0.0

    def send_msg_stop(self):
        #self.get_logger().info(f'send_msg_stop function was entered')
        msg = DutyCycles()
        msg.duty_cycle_left = 0
        msg.duty_cycle_right = 0
        self.pub.publish(msg)

    def send_msg_change_vel(self, left, right):
        self.left_duty_cycle += left
        self.right_duty_cycle += right

        # Clamp the duty cycles to the valid range
        self.left_duty_cycle = max(DUTY_CYCLE_MIN, min(DUTY_CYCLE_MAX, self.left_duty_cycle))
        self.right_duty_cycle = max(DUTY_CYCLE_MIN, min(DUTY_CYCLE_MAX, self.right_duty_cycle))

        msg = DutyCycles()
        msg.duty_cycle_left = self.left_duty_cycle
        msg.duty_cycle_right = self.right_duty_cycle
        self.pub.publish(msg)

    def on_press(self, key):
        #print(f'Key {key.char} pressed')
        #self.get_logger().info(f'send_msg_stop function was entered')
        try:
            if key.char == 'q':
                self.send_msg_stop()
            elif key.char == 'w':
                self.send_msg_change_vel(0.1, 0.1)
            elif key.char == 'a':
                self.send_msg_change_vel(0.1, 0)
            elif key.char == 'd':
                self.send_msg_change_vel(0, 0.1)
            elif key.char == 's':
                self.send_msg_change_vel(-0.1, -0.1)

        except AttributeError:
            pass
            

def main():
    rclpy.init()
    node = Driver()
    #rclpy.spin_once(node)
    listener = keyboard.Listener(
        on_press=node.on_press)#,
        #on_release=on_release)
    listener.start() #on a seperate thread (non-blocking)
    rclpy.spin(node)
    #rclpy.spin_once(node)

    # while(True):
    #      continue


    rclpy.shutdown()

if __name__ == '__main__':
    main()


# def on_release(key):
#     print(f'Key {key.char} released')
#     if key == keyboard.Key.esc:
#         # Stop listener
#         return False
