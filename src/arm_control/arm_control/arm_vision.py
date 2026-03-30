import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray


MIN_CONTOUR_AREA = 500.0
IMAGE_TOPIC = '/arm/camera/image_raw'
GREEN_CENTER_TOPIC = '/arm/vision/green_cube_center'
DEBUG_IMAGE_TOPIC = '/arm/vision/debug_image'
MASK_TOPIC_TEMPLATE = '/arm/vision/{color}_mask'

COLOR_RANGES = {
    'green': {
        'lower': np.array([40, 40, 40], dtype=np.uint8),
        'upper': np.array([80, 255, 255], dtype=np.uint8),
        'box_color': (0, 255, 0),
    },
    'red': {
        'ranges': [
            {
                'lower': np.array([0, 120, 70], dtype=np.uint8),
                'upper': np.array([7, 255, 255], dtype=np.uint8),
            },
            {
                'lower': np.array([170, 120, 70], dtype=np.uint8),
                'upper': np.array([180, 255, 255], dtype=np.uint8),
            }
        ],
        'box_color': (0, 0, 255),
    },
    'blue': {
        'lower': np.array([100, 80, 40], dtype=np.uint8),
        'upper': np.array([130, 255, 255], dtype=np.uint8),
        'box_color': (255, 0, 0),
    },
}


class ArmVisionNode(Node):
    def __init__(self):
        super().__init__('arm_vision')

        self.bridge = CvBridge()
        self.min_contour_area = MIN_CONTOUR_AREA
        self.center_publishers = {
            'green': self.create_publisher(Int32MultiArray, GREEN_CENTER_TOPIC, 10),
        }
        self.mask_publishers = {
            color: self.create_publisher(Image, MASK_TOPIC_TEMPLATE.format(color=color), 10)
            for color in COLOR_RANGES
        }
        self.debug_image_pub = self.create_publisher(Image, DEBUG_IMAGE_TOPIC, 10)

        self.image_subscription = self.create_subscription(
            Image,
            IMAGE_TOPIC,
            self.image_callback,
            10,
        )

    def image_callback(self, msg: Image):
        frame = self._ros_image_to_bgr(msg)
        if frame is None:
            return

        debug_image = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for color_name, color_config in COLOR_RANGES.items():
            detection = self.detect_cube(hsv, color_name, color_config)
            self.mask_publishers[color_name].publish(
                self.bridge.cv2_to_imgmsg(detection['mask'], encoding='mono8')
            )

            if detection['center'] is None:
                continue

            self.draw_detection(debug_image, color_name, detection, color_config['box_color'])

            if color_name in self.center_publishers:
                center_msg = Int32MultiArray()
                center_msg.data = [detection['center'][0], detection['center'][1]]
                self.center_publishers[color_name].publish(center_msg)

        self.debug_image_pub.publish(self.bridge.cv2_to_imgmsg(debug_image, encoding='bgr8'))

    def detect_cube(self, hsv, color_name, color_config):
        if 'ranges' in color_config:
            mask = None
            for r in color_config['ranges']:
                m = cv2.inRange(hsv, r['lower'], r['upper'])
                mask = m if mask is None else (mask | m)
        else:
            mask = cv2.inRange(hsv, color_config['lower'], color_config['upper'])
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return {'mask': mask, 'center': None, 'bbox': None}

        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        if contour_area < self.min_contour_area:
            return {'mask': mask, 'center': None, 'bbox': None}

        moments = cv2.moments(largest_contour)
        if moments['m00'] == 0:
            return {'mask': mask, 'center': None, 'bbox': None}

        center_x = int(moments['m10'] / moments['m00'])
        center_y = int(moments['m01'] / moments['m00'])
        bbox = cv2.boundingRect(largest_contour)

        return {
            'mask': mask,
            'center': (center_x, center_y),
            'bbox': bbox,
        }

    def draw_detection(self, image, color_name, detection, box_color):
        x, y, width, height = detection['bbox']
        center_x, center_y = detection['center']

        cv2.rectangle(image, (x, y), (x + width, y + height), box_color, 2)
        cv2.circle(image, (center_x, center_y), 6, (255, 255, 255), -1)
        cv2.circle(image, (center_x, center_y), 4, box_color, -1)
        cv2.putText(
            image,
            f'{color_name} cube ({center_x}, {center_y})',
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            box_color,
            2,
            cv2.LINE_AA,
        )

    def _ros_image_to_bgr(self, msg: Image):
        if msg.encoding == 'bgr8':
            return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)

        if msg.encoding == 'rgb8':
            rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if msg.encoding == 'yuv422_yuy2':
            yuy = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 2))
            return cv2.cvtColor(yuy, cv2.COLOR_YUV2BGR_YUY2)

        self.get_logger().warn(f'Encoding {msg.encoding} not supported')
        return None


def main():
    rclpy.init()
    node = ArmVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
