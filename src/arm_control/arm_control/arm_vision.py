import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray
import colour as co
from std_msgs.msg import String

MIN_CONTOUR_AREA = 450.0
MIN_HOLDING_WIDTH = 130
HOLDING_TIMEOUT_SEC = 2.0

#TOPICS
IMAGE_TOPIC = '/arm/camera/image_raw'
CENTER_TOPIC = '/arm/vision/cube_center'
DEBUG_IMAGE_TOPIC = '/arm/vision/debug_image'
MASK_TOPIC_TEMPLATE = '/arm/vision/{color}_mask'
HOLDING_CHECK_TOPIC = '/arm/vision/holding_check'
HOLDING_ANSWER_TOPIC = '/arm/vision/holding_answer'

#Holding messages
CHECK_HOLDING_MSG = 'CHECK_HOLDING'
HOLDING_SUCCESS_MSG = 'HOLDING_SUCCESS'
HOLDING_FAIL_MSG = 'HOLDING_FAIL'

#DEBUG
DEBUG_COLOR_PICKER_ENABLED = False
DEBUG_PUBLISH_MASKS = False

ARM_COLORS_RGB = {
    'green': np.array([0, 255, 0]),
    'red': np.array([255, 0, 0]),
    'blue': np.array([0, 230, 255])
}

TOLERANCES = {
    'green': 0.25,
    'red': 0.21,
    'blue': 0.105
}

L_BOUNDS = {
    "red": (0.2, 0.85),
    "green": (0.1, 0.8),
    "blue": (0.2, 0.9)
}

BOX_COLORS = {
    'green': (0, 255, 0),
    'red': (0, 0, 255),
    'blue': (255, 0, 0)
}

CANNY_THRESH = {
    'low': 30,
    'high': 80,
}

class ArmVisionNode(Node):
    def __init__(self):
        super().__init__('arm_vision')

        self.holding = False

        self.smoothed_angles = {color: None for color in ARM_COLORS_RGB}

        self.bridge = CvBridge()

        self.oklab_refs = {}
        for color, rgb in ARM_COLORS_RGB.items():
            rgb_norm = rgb / 255.0
            xyz = co.sRGB_to_XYZ(rgb_norm)
            self.oklab_refs[color] = co.XYZ_to_Oklab(xyz)
        self.tolerances = TOLERANCES
        self.box_colors = BOX_COLORS

        self.center_publisher = self.create_publisher(Int32MultiArray, CENTER_TOPIC, 10)
        if DEBUG_PUBLISH_MASKS:
            self.mask_publishers = {
                color: self.create_publisher(Image, MASK_TOPIC_TEMPLATE.format(color=color), 10)
                for color in ARM_COLORS_RGB
            }

            self.edge_mask_publishers = {
                color: self.create_publisher(Image, MASK_TOPIC_TEMPLATE.format(color=color) + '_edges', 10)
                for color in ARM_COLORS_RGB
            }

            self.debug_image_pub = self.create_publisher(Image, DEBUG_IMAGE_TOPIC, 10)

        self.image_subscription = self.create_subscription(
            Image,
            IMAGE_TOPIC,
            self.image_callback_color,
            10,
        )

        self.holding_pub = self.create_publisher(String, HOLDING_ANSWER_TOPIC, 10)
        self.create_subscription(String, HOLDING_CHECK_TOPIC, self.holding_callback, 10)

    def holding_callback(self, msg):
        if msg.data == CHECK_HOLDING_MSG:
            self.holding = True
            self.holding_timer = self.create_timer(
                HOLDING_TIMEOUT_SEC, self._holding_timeout
            )

    def _holding_timeout(self):
        if self.holding:
            result = String()
            result.data = HOLDING_FAIL_MSG
            self.holding_pub.publish(result)
            self.holding = False

        if self.holding_timer is not None:
            self.holding_timer.cancel()
            self.holding_timer = None

    def image_callback_color(self, msg: Image):
        frame = self._ros_image_to_bgr(msg)
        if frame is None:
            return

        if self.holding:
            h, w = frame.shape[:2]
            frame = frame[h // 2:, int(w * 0.3):int(w * 0.8)]

        if DEBUG_COLOR_PICKER_ENABLED:
            cv2.imshow("debug", frame)
            key = cv2.waitKey(1)
            if key == ord('p'):
                self.debug_color_picker(frame)

        debug_image = frame.copy()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        xyz = co.sRGB_to_XYZ(rgb)
        oklab = co.XYZ_to_Oklab(xyz)

        detections = []
        for color_name in ARM_COLORS_RGB.keys():
            detection = self.detect_cube_color(oklab, color_name)
            if DEBUG_PUBLISH_MASKS:
                self.mask_publishers[color_name].publish(
                    self.bridge.cv2_to_imgmsg(detection['mask'], encoding='mono8')
                )

            if DEBUG_PUBLISH_MASKS and detection['edges'] is not None:
                self.edge_mask_publishers[color_name].publish(
                    self.bridge.cv2_to_imgmsg(detection['edges'], encoding='mono8')
                )

            if detection['center'] is None:
                continue

            detections.append({'detection': detection, 'color': color_name, 'confidence': detection['confidence']})

            if self.holding:
                if detection['box_w'] >= MIN_HOLDING_WIDTH:
                    msg = String()
                    msg.data = HOLDING_SUCCESS_MSG
                    self.holding_pub.publish(msg)
                    self.holding = False

        if detections:
            best = max(detections, key=lambda x: x['confidence'])
            best_detection = best['detection']
            best_color = best['color']

            if DEBUG_PUBLISH_MASKS:
                self.draw_detection(debug_image, best_detection, self.box_colors[best_color])

            center_msg = Int32MultiArray()
            center_msg.data = [
                best_detection['center'][0],
                best_detection['center'][1],
                int(best_detection['angle']),  
            ]
            self.center_publisher.publish(center_msg)

        if DEBUG_PUBLISH_MASKS:
            self.debug_image_pub.publish(self.bridge.cv2_to_imgmsg(debug_image, encoding='bgr8'))

    def chroma_edges(self, oklab, color_name):
        ref = self.oklab_refs[color_name]
        tol = self.tolerances[color_name]

        a = oklab[:, :, 1]
        b = oklab[:, :, 2]

        color_dist = np.sqrt((a - ref[1])**2 + (b - ref[2])**2)

        chroma_confidence = 1.0 - color_dist / tol
        chroma_confidence = np.clip(chroma_confidence, 0, 1)
        chroma_u8 = (chroma_confidence * 255).astype(np.uint8)

        chroma_blurred = cv2.GaussianBlur(chroma_u8, (3, 3), 0)
        edges = cv2.Canny(chroma_blurred, CANNY_THRESH['low'], CANNY_THRESH['high'])

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        return edges

    def detect_cube_color(self, oklab, color_name):
        ref = self.oklab_refs[color_name]
        tol = self.tolerances[color_name]

        L = oklab[:, :, 0]
        a = oklab[:, :, 1]
        b = oklab[:, :, 2]

        L_min, L_max = L_BOUNDS[color_name]
        color_dist = (a - ref[1])**2 + (b - ref[2])**2
        mask = (color_dist < tol**2) & (L_min < L) & (L < L_max)
        mask = mask.astype(np.uint8) * 255

        edges = self.chroma_edges(oklab, color_name)
        edge_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        color_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = edge_contours if edge_contours else color_contours

        if not contours:
            self.smoothed_angles[color_name] = None 
            return {'mask': mask, 'center': None, 'bbox': None, 'box_w': 0, 'angle': 0, 'box_points': None, 'edges': edges}

        valid_contours = [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]

        if not valid_contours:
            self.smoothed_angles[color_name] = None 
            return {'mask': mask, 'center': None, 'bbox': None, 'box_w': 0, 'angle': 0, 'box_points': None, 'edges': edges}

        best_contour = max(valid_contours, key=self.squareness)
        if self.squareness(best_contour) < 0.4:
            self.smoothed_angles[color_name] = None 
            return {'mask': mask, 'center': None, 'bbox': None, 'box_w': 0, 'angle': 0, 'box_points': None, 'edges': edges}

        largest_contour = best_contour
        rect = cv2.minAreaRect(largest_contour)
        center = (int(rect[0][0]), int(rect[0][1]))
        box_w, box_h = rect[1]
        angle = rect[2]

        angle = angle % 90
        if angle > 45:
            angle -= 90

        alpha = 0.4
        prev = self.smoothed_angles[color_name]
        if prev is None:
            self.smoothed_angles[color_name] = angle
        else:
            diff = angle - prev
            if diff > 45:
                diff -= 90
            elif diff < -45:
                diff += 90
            self.smoothed_angles[color_name] = np.clip(prev + alpha * diff, -45, 45)


        angle = self.smoothed_angles[color_name]

        box_points = cv2.boxPoints(rect).astype(np.int32)

        return {
            'mask': mask,
            'center': center,
            'bbox': cv2.boundingRect(largest_contour),  
            'box_w': box_w,          
            'angle': angle,          
            'box_points': box_points, 
            'edges': edges,
            'confidence': self.squareness(best_contour),
        }
    
    def squareness(self,contour):
        _, (w, h), _ = cv2.minAreaRect(contour)
        if max(w, h) == 0:
            return 0
        return min(w, h) / max(w, h)

    def draw_detection(self, image, detection, box_color):
        center_x, center_y = detection['center']
        angle = detection['angle']

        cv2.drawContours(image, [detection['box_points']], 0, box_color, 2)

        length = 40
        end_x = int(center_x + length * np.cos(np.radians(angle)))
        end_y = int(center_y - length * np.sin(np.radians(angle)))
        cv2.arrowedLine(image, (center_x, center_y), (end_x, end_y), (255, 255, 0), 2)

        cv2.circle(image, (center_x, center_y), 5, box_color, -1)
        cv2.putText(
            image,
            f'cube ({center_x}, {center_y}) {angle:.1f}deg',  
            (center_x + 8, center_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            box_color,
            1,
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

    def debug_color_picker(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        xyz = co.sRGB_to_XYZ(rgb)
        oklab = co.XYZ_to_Oklab(xyz)

        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                roi = oklab[max(0,y-2):y+3, max(0,x-2):x+3]
                L = roi[:,:,0].mean()
                a = roi[:,:,1].mean()
                b = roi[:,:,2].mean()
                roi_bgr = frame[max(0,y-2):y+3, max(0,x-2):x+3]
                rgb = roi_bgr[:,:,::-1].mean(axis=(0,1)).astype(int)
                print(f"Clicked ({x}, {y}) → RGB=({rgb[0]}, {rgb[1]}, {rgb[2]}) | L={L:.3f}, a={a:.3f}, b={b:.3f}")
                for color_name, ref in self.oklab_refs.items():
                    dist = np.sqrt((a - ref[1])**2 + (b - ref[2])**2)
                    L_min, L_max = L_BOUNDS[color_name]
                    in_L = L_min < L < L_max
                    print(f"  {color_name}: chroma_dist={dist:.3f} (tol={self.tolerances[color_name]}), L_in_bounds={in_L}")

        cv2.namedWindow("debug")
        cv2.setMouseCallback("debug", on_click)
        cv2.imshow("debug", frame)
        cv2.waitKey(0)


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