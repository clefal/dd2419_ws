import serial
import struct

import rclpy
from rclpy.node import Node

from robp_interfaces.msg import ArmControl, ArmFeedback
from std_msgs.msg import Empty

class Arm(Node):
  def __init__(self):
    super().__init__("arm")

    self.ser = serial.Serial("/dev/hiwonder_arm", baudrate=115200)

    self.pub = self.create_publisher(ArmFeedback, "/arm/feedback", 10)
    self.sub = self.create_subscription(ArmControl, "/arm/control", self.arm_control, 10)
    self.reset_sub = self.create_subscription(Empty, "/arm/reset", self.arm_reset, 10)

    self.last_msg = None
    self.cur_msg = None

    self.reset = True

  def arm_control(self, msg: ArmControl) -> None:
    self.cur_msg = msg

  def arm_reset(self, msg: Empty) -> None:
    self.reset = True


  def feedback(self) -> None:
    action = 0b010


    if self.reset:
      self.last_msg = None
      self.cur_msg = None
      self.reset = False
      action |= 0b100
    elif self.cur_msg and (not self.last_msg or (self.last_msg.position != self.cur_msg.position).any()):
      self.last_msg = self.cur_msg
      action |= 0b001

    self.ser.write(struct.pack('<b', action))

    if 0b001 & action:
      self.ser.write(struct.pack(f'<{len(self.cur_msg.position)}f', *self.cur_msg.position))
      self.ser.write(struct.pack(f'<{len(self.cur_msg.time)}H', *self.cur_msg.time))
    
    msg = ArmFeedback()

    self.ser.read_until(b"BEGIN FEEDBACK")
    data = self.ser.read_until(b"END FEEDBACK")[:4*6]
    msg.position = struct.unpack('<6f', data)

    msg.header.frame_id = "arm_link"
    msg.header.stamp = self.get_clock().now().to_msg()

    self.pub.publish(msg)

def main() -> None:
  rclpy.init()
  arm = Arm()

  while rclpy.ok():
    arm.feedback()
    rclpy.spin_once(arm, timeout_sec=0.001)

if __name__ == '__main__':
    main()