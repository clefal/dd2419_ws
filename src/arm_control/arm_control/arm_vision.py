import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray
import colour as co

#TODO OPENCV EDGE DETECTION
#TODO CAMERA CALIBRATION

# TODO ISTURNING ON CONTROLLER; HIGH W DELTAS

# TODO NO OBJECTS IN 
#TODO: Replanning:
# PATHMANAGER: checks live planning grid against planned path
# on interception replan. on new object detected replan

#TODO EXPLORATION PHASE (GO INTO SMALL PART OF WORKSPACE!)


MIN_CONTOUR_AREA = 500.0
IMAGE_TOPIC = '/arm/camera/image_raw'
CENTER_TOPIC = '/arm/vision/cube_center'
DEBUG_IMAGE_TOPIC = '/arm/vision/debug_image'
MASK_TOPIC_TEMPLATE = '/arm/vision/{color}_mask'

ARM_COLORS_RGB = {
    'green': np.array([0, 255, 0]),
    'red': np.array([255, 0, 0]),    #rgba(243, 140, 173)
    'blue': np.array([0, 0, 255])
}

TOLERANCES = {
    'green': 0.25,
    'red': 0.185,
    'blue': 0.28
}

L_BOUNDS = {
    "red": (0.2, 0.8),
    "green": (0.2, 0.8),
    "blue": (0.2, 0.81)
}

BOX_COLORS = {
    'green': (0, 255, 0),
    'red': (0, 0, 255),
    'blue': (255, 0, 0)
}


class ArmVisionNode(Node):
    def __init__(self):
        super().__init__('arm_vision')

        self.bridge = CvBridge()

        # Compute Oklab references
        self.oklab_refs = {}
        for color, rgb in ARM_COLORS_RGB.items():
            rgb_norm = rgb / 255.0
            xyz = co.sRGB_to_XYZ(rgb_norm)
            self.oklab_refs[color] = co.XYZ_to_Oklab(xyz)
        self.tolerances = TOLERANCES
        self.box_colors = BOX_COLORS

        self.min_contour_area = MIN_CONTOUR_AREA
        self.center_publisher =  self.create_publisher(Int32MultiArray, CENTER_TOPIC, 10)
        self.mask_publishers = {
            color: self.create_publisher(Image, MASK_TOPIC_TEMPLATE.format(color=color), 10)
            for color in ARM_COLORS_RGB
        }

        self.mask_publisher = self.create_publisher(Image, '/arm/vision/edge_mask', 10)

        self.debug_image_pub = self.create_publisher(Image, DEBUG_IMAGE_TOPIC, 10)

        self.image_subscription = self.create_subscription(
            Image,
            IMAGE_TOPIC,
            self.image_callback_color,
            10,
        )

        # self.image_subscription = self.create_subscription(
        #     Image,
        #     IMAGE_TOPIC,
        #     self.image_callback_edges,
        #     10,
        # )

    def image_callback_color(self, msg: Image):
        frame = self._ros_image_to_bgr(msg)
        if frame is None:
            return
        
        # cv2.imshow("debug", frame)
        # key = cv2.waitKey(1)
        
        # if key == ord('p'):  # press P to pause and click around
        #     self.debug_color_picker(frame)

        debug_image = frame.copy()

        # Convert to Oklab
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        xyz = co.sRGB_to_XYZ(rgb)
        oklab = co.XYZ_to_Oklab(xyz)

        for color_name in ARM_COLORS_RGB.keys():
            detection = self.detect_cube_color(oklab, color_name)
            self.mask_publishers[color_name].publish(
                self.bridge.cv2_to_imgmsg(detection['mask'], encoding='mono8')
            )

            if detection['center'] is None:
                continue

            self.draw_detection(debug_image, detection, self.box_colors[color_name])

            center_msg = Int32MultiArray()
            center_msg.data = [detection['center'][0], detection['center'][1]]
            self.center_publisher.publish(center_msg)

        self.debug_image_pub.publish(self.bridge.cv2_to_imgmsg(debug_image, encoding='bgr8'))

    def image_callback_edges(self, msg: Image):
        frame = self._ros_image_to_bgr(msg)
        if frame is None:
            return

        debug_image = frame.copy()

        detection = self.detect_cube_edges(frame)
        self.mask_publisher.publish(
            self.bridge.cv2_to_imgmsg(detection['mask'], encoding='mono8')
        )

        if detection['center'] is not None:
            self.draw_detection(debug_image, detection, (0, 255, 0))  # single fixed colour for bounding box

            center_msg = Int32MultiArray()
            center_msg.data = [detection['center'][0], detection['center'][1]]
            self.center_publisher.publish(center_msg)

        self.debug_image_pub.publish(self.bridge.cv2_to_imgmsg(debug_image, encoding='bgr8'))

    def detect_cube_color(self, oklab, color_name):
        ref = self.oklab_refs[color_name]
        tol = self.tolerances[color_name]
        
        L = oklab[:, :, 0]
        a = oklab[:, :, 1]
        b = oklab[:, :, 2]

        L_min, L_max = L_BOUNDS[color_name]
        chroma_tol = tol
        #chroma_tol = tol * (1.0 + 1.5 * L)  # tolerance grows with brightness
        color_dist = (a - ref[1])**2 + (b - ref[2])**2
        mask = (color_dist < chroma_tol**2) & (L_min < L) & (L < L_max)

        mask = mask.astype(np.uint8) * 255
        
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
    
    def detect_cube_edges(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_contour = None
        best_score = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

            if len(approx) != 4:
                continue

            _, _, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / h
            if not (0.6 < aspect_ratio < 1.6):
                continue

            #Determine which contour is best by combining how square it is and how well it fills in 
            squareness = min(w, h) / max(w, h)
            fill_ratio = area / (w * h)
            score = squareness * fill_ratio

            if score > best_score:
                best_score = score
                best_contour = approx

        if best_contour is None:
            return {'mask': edges, 'center': None, 'bbox': None}

        moments = cv2.moments(best_contour)
        if moments['m00'] == 0:
            return {'mask': edges, 'center': None, 'bbox': None}

        center_x = int(moments['m10'] / moments['m00'])
        center_y = int(moments['m01'] / moments['m00'])
        bbox = cv2.boundingRect(best_contour)

        return {'mask': edges, 'center': (center_x, center_y), 'bbox': bbox}

    def draw_detection(self, image, detection, box_color):
        x, y, width, height = detection['bbox']
        center_x, center_y = detection['center']

        cv2.rectangle(image, (x, y), (x + width, y + height), box_color, 2)
        cv2.circle(image, (center_x, center_y), 6, (255, 255, 255), -1)
        cv2.circle(image, (center_x, center_y), 4, box_color, -1)
        cv2.putText(
            image,
            f'cube ({center_x}, {center_y})',
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
    
    def debug_color_picker(self, frame):
        """Click on any pixel to print its OKLab values"""
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        xyz = co.sRGB_to_XYZ(rgb)
        oklab = co.XYZ_to_Oklab(xyz)

        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                roi = oklab[max(0,y-2):y+3, max(0,x-2):x+3]
                L = roi[:,:,0].mean()
                a = roi[:,:,1].mean()
                b = roi[:,:,2].mean()
                
                # Sample RGB from original frame
                roi_bgr = frame[max(0,y-2):y+3, max(0,x-2):x+3]
                rgb = roi_bgr[:,:,::-1].mean(axis=(0,1)).astype(int)  # flip BGR→RGB
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
