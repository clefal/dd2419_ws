#!/usr/bin/env python
import numpy as np
import colour as co
import rclpy
from rclpy.node import Node
import rclpy.duration

from sklearn.cluster import DBSCAN

from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointField
from geometry_msgs.msg import PointStamped
from tf2_ros import Buffer, TransformListener
from scipy.spatial.transform import Rotation
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
import math


class Detection(Node):

    def __init__(self):
        super().__init__('detection')

        ### DECLARE PARAMETERS ###          
        # Clustering & Detection params
        self.declare_parameter("min_samples", 10)
        self.declare_parameter("eps", 0.03)
        self.declare_parameter("obj_width", 0.03)
        self.declare_parameter("box_min_width", 0.09)
        self.declare_parameter("box_max_width", 0.28)
        self.declare_parameter("obj_tolerance", 0.03)
        self.declare_parameter("buffer_size", 3)
        self.declare_parameter("max_general_counter", 6000)
        self.declare_parameter("obstacle_distance_m", 0.15)
        self.declare_parameter("occupancy_threshold", 51) # threshold used for occupancy grid check

        # Topic params
        self.declare_parameter("input_cloud_topic", "/realsense/depth/color/points")
        self.declare_parameter("output_pointcloud_topic", "/camera/depth/color/ds_points")
        self.declare_parameter("red_cube_topic", "/detection/objects/red_cube")
        self.declare_parameter("green_cube_topic", "/detection/objects/green_cube")
        self.declare_parameter("blue_cube_topic", "/detection/objects/blue_cube")
        self.declare_parameter("wood_cube_topic", "/detection/objects/wood_cube")
        self.declare_parameter("box_topic", "/detection/objects/box")
        self.declare_parameter("occupancy_grid_topic", "/map/occupancy_grid")

        ### GET PARAMETERS ###
        # Set clustering & detection params to class attributes
        self.min_samples = self.get_parameter("min_samples").value
        self.eps = self.get_parameter("eps").value
        self.obj_width = self.get_parameter("obj_width").value
        self.obj_tolerance = self.get_parameter("obj_tolerance").value
        self.box_min_width = self.get_parameter("box_min_width").value
        self.box_max_width = self.get_parameter("box_max_width").value
        self.buffer_size = self.get_parameter("buffer_size").value
        self.max_general_counter = self.get_parameter("max_general_counter").value

        # Fetch topic strings 
        input_cloud_topic = self.get_parameter("input_cloud_topic").value
        output_pointcloud_topic = self.get_parameter("output_pointcloud_topic").value
        red_cube_topic = self.get_parameter("red_cube_topic").value
        green_cube_topic = self.get_parameter("green_cube_topic").value
        blue_cube_topic = self.get_parameter("blue_cube_topic").value
        wood_cube_topic = self.get_parameter("wood_cube_topic").value
        box_topic = self.get_parameter("box_topic").value
        occupancy_grid_topic = self.get_parameter("occupancy_grid_topic").value

        # ------------------- INTERNAL SETUP ---------------
        # get thresholds during initialization
        self.thresh = self.get_thresholds()

        # initialize point buffering
        self.point_buffers = {'red': [], 'green':[], 'blue': [], 'wood':[], 'box':[]}

        # initialize TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Initialize the publisher
        self._pub = self.create_publisher(
            PointCloud2, output_pointcloud_topic, 10, callback_group=ReentrantCallbackGroup())
        
        self.red_centroid_pub = self.create_publisher(PointStamped, red_cube_topic, 10, callback_group=ReentrantCallbackGroup())
        self.green_centroid_pub = self.create_publisher(PointStamped, green_cube_topic, 10, callback_group=ReentrantCallbackGroup())
        self.blue_centroid_pub = self.create_publisher(PointStamped, blue_cube_topic, 10, callback_group=ReentrantCallbackGroup())
        self.wood_centroid_pub = self.create_publisher(PointStamped, wood_cube_topic, 10, callback_group=ReentrantCallbackGroup())
        self.box_centroid_pub = self.create_publisher(PointStamped, box_topic, 10, callback_group=ReentrantCallbackGroup())

        # Subscribe to point cloud topic and call callback function on each received message
        self.create_subscription(
            PointCloud2, input_cloud_topic, self.cloud_callback, 10, callback_group=ReentrantCallbackGroup())
        
        # Define the Latched QoS Profile for the Occupancy Grid Subscription
        latched_qos = QoSProfile(
            depth=1,                                            # Keep only the last message
            history=QoSHistoryPolicy.KEEP_LAST,                 # Standard history policy for latching
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,     # THIS is what makes it "latched"
            reliability=QoSReliabilityPolicy.RELIABLE           # Ensure the latched message actually arrives
        )
        self.create_subscription(OccupancyGrid, occupancy_grid_topic, self.occupancy_grid_callback, callback_group=ReentrantCallbackGroup(),qos_profile=latched_qos)
        self.occupancy_grid = None 

    def cloud_callback(self, msg: PointCloud2):
        """
        Takes point cloud readings to detect objects.
        This function is called for every message that is published on the '/camera/depth/color/points' topic.
        """

        # convert pointcloud to numpy arrays
        gen = pc2.read_points_numpy(msg, skip_nans=True)
        points = gen[:, :3]
        rgb_uint32 = gen[:, 3].view(np.uint32)
        colors = np.empty((len(rgb_uint32), 3), dtype=np.uint8)
        colors[:, 0] = (rgb_uint32 >> 16) & 255
        colors[:, 1] = (rgb_uint32 >> 8) & 255
        colors[:, 2] = rgb_uint32 & 255



        # geometrical filter
        # these thresholds are applied in the camera frame, that is why handling them can be counter intuitive
        max_dist = 2
        max_height = 0.05   
        min_height = 0.08
        geom_mask = ((points[:,2] < max_dist) & (points[:,1] > max_height) & (points[:,1] < min_height))
        points_f = points[geom_mask]
        colors_f = colors[geom_mask]

        max_dist_box = 2
        max_height_box = 0.00   
        min_height_box = 0.05
        geom_mask_for_box =  ((points[:,2] < max_dist_box) & (points[:,1] > max_height_box) & (points[:,1] < min_height_box))
        points_f_box = points[geom_mask_for_box]
        colors_f_box = colors[geom_mask_for_box]



        # transform points to map coordinates
        points_map = self.transform_points_to_map(points_f, msg.header)
        points_map_box = self.transform_points_to_map(points_f_box, msg.header)



        # get masks and check how many hits we have in general 
        red_mask, green_mask, blue_mask, wood_mask, box_mask = self.get_masks(colors_f, colors_f_box) # returns the color masks based on threshold values

        red_counter = np.sum(red_mask)
        green_counter = np.sum(green_mask)
        blue_counter = np.sum(blue_mask)
        wood_counter = np.sum(wood_mask)
        box_counter = np.sum(box_mask)

        general_counter = red_counter + green_counter + blue_counter + wood_counter + box_counter



        # end callback if we have no hits in general
        if general_counter == 0: return 
        
        # end callback if we detect too many colorful points (because it is likely that there are a lot of false positives)
        elif general_counter >= self.max_general_counter: 
            self.get_logger().info(f'many hits {general_counter} by color thresholding, danger of false positives, detection iteration aborted')
            self.point_buffers['red'] = []
            self.point_buffers['green'] = []
            self.point_buffers['blue'] = []
            self.point_buffers['wood'] = []
            self.point_buffers['box'] = []
            return


        # needed to publish te pointcloud for rviz
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            ]
        centroid_header = Header()
        centroid_header.stamp = msg.header.stamp  #this is a bit sus, since we are buffering the points
        centroid_header.frame_id = 'map'


        # If points are converted successfully then add them to buffer, if the buffer is full then run clustering and remove the oldest points in the buffer
        # repeat tht for every color
        if points_map.shape == points_f.shape:  # this is only the case if the transform_points_to_map actually succeeds
            
            # manage red_points
            if red_counter > 0: # add points to buffer if we have more than a minimum amount of hits
                red_points = points_map[red_mask]
                self.point_buffers['red'].append(red_points)
            
            if red_counter == 0 and len(self.point_buffers['red'])!= 0 : # clear the buffer if we dont see points in consecutive scans
                self.point_buffers['red'] = [] 

            if len(self.point_buffers['red'])>=self.buffer_size:
                all_red_points = np.vstack(self.point_buffers['red'])
                red_centroids = self.process_clusters(all_red_points)
                # self.get_logger().info(f'red: {len(all_red_points)}')

                # only for visualization in rviz
                msg_red = pc2.create_cloud(centroid_header, fields, all_red_points)
                self._pub.publish(msg_red)

                for centroid in red_centroids: 
                    self.publish_detection(centroid, centroid_header, 'red')

                del self.point_buffers['red'][0] # after publishing clear oldes points of the buffer
            

            # manage green_points
            if green_counter > 0: # add points to buffer if we have more than a minimum amount of hits
                green_points = points_map[green_mask]
                self.point_buffers['green'].append(green_points)

            if green_counter == 0 and len(self.point_buffers['green'])!= 0 : # clear the buffer if we dont see points in consecutive scans
                self.point_buffers['green'] = [] 

            if len(self.point_buffers['green'])>=self.buffer_size:
                all_green_points = np.vstack(self.point_buffers['green'])
                green_centroids = self.process_clusters(all_green_points)
                # self.get_logger().info(f'green: {len(all_green_points)}')

                # only for visualization in rviz
                msg_green = pc2.create_cloud(centroid_header, fields, all_green_points)
                self._pub.publish(msg_green)

                for centroid in green_centroids: 
                    self.publish_detection(centroid, centroid_header, 'green')
                    
                del self.point_buffers['green'][0] # after publishing clear oldes points of the buffer


            # manage blue_points
            if blue_counter > 0: # add points to buffer if we have more than a minimum amount of hits
                blue_points = points_map[blue_mask]
                self.point_buffers['blue'].append(blue_points)

            if blue_counter == 0 and len(self.point_buffers['blue'])!= 0 : # clear the buffer if we dont see points in consecutive scans
                self.point_buffers['blue'] = [] 

            if len(self.point_buffers['blue'])>=self.buffer_size:
                all_blue_points = np.vstack(self.point_buffers['blue'])
                blue_centroids = self.process_clusters(all_blue_points)
                # self.get_logger().info(f'blue: {len(all_blue_points)}')

                # only for visualization in rviz
                msg_blue = pc2.create_cloud(centroid_header, fields, all_blue_points)
                self._pub.publish(msg_blue)

                for centroid in blue_centroids: 
                    self.publish_detection(centroid, centroid_header, 'blue')
                    
                del self.point_buffers['blue'][0] # after publishing clear oldes points of the buffer


            # manage wood_points
            if wood_counter > 0: # add points to buffer if we have more than a minimum amount of hits
                wood_points = points_map[wood_mask]
                self.point_buffers['wood'].append(wood_points)

            if wood_counter == 0 and len(self.point_buffers['wood'])!= 0 : # clear the buffer if we dont see points in consecutive scans
                self.point_buffers['wood'] = [] 

            if len(self.point_buffers['wood'])>=self.buffer_size:
                all_wood_points = np.vstack(self.point_buffers['wood'])
                wood_centroids = self.process_clusters(all_wood_points)
                # self.get_logger().info(f'wood: {len(all_wood_points)}')

                # only for visualization in rviz
                msg_wood = pc2.create_cloud(centroid_header, fields, all_wood_points)
                self._pub.publish(msg_wood)

                #for centroid in wood_centroids:    
                    # self.publish_detection(centroid, centroid_header, 'wood')
                    
                del self.point_buffers['wood'][0] # after publishing clear oldes points of the buffer
                        

            # manage box points
            if box_counter > 0: # add points to buffer if we have more than a minimum amount of hits
                box_points = points_map_box[box_mask]
                self.point_buffers['box'].append(box_points)

            if box_counter == 0 and len(self.point_buffers['box'])!= 0 : # clear the buffer if we dont see points in consecutive scans
                self.point_buffers['box'] = [] 

            if len(self.point_buffers['box'])>=self.buffer_size:
                all_box_points = np.vstack(self.point_buffers['box'])
                box_centroids = self.process_clusters(all_box_points, box=True)
                # self.get_logger().info(f'box: {len(all_box_points)}')

                # only for visualization in rviz
                msg_box = pc2.create_cloud(centroid_header, fields, all_box_points)
                self._pub.publish(msg_box)

                for centroid in box_centroids:
                    self.publish_detection(centroid, centroid_header, 'box')
                    
                del self.point_buffers['box'][0] # after publishing clear oldes points of the buffer
            
    def occupancy_grid_callback(self, msg :OccupancyGrid):
        #self.get_logger().info(f'revieved occupancy grid message')
        self.occupancy_grid = msg
        return

    def transform_points_to_map(self, points_np, header :Header):
        """
        Converts an (N, 3) numpy array of points from camera frame to map frame.
        """
        source_frame = header.frame_id
        target_frame = 'map'
        stamp = header.stamp
        if len(points_np) == 0:
            self.get_logger().warn(f'transform_points_to_map() had an empty point array as input')
            return np.empty((0,3))
        
        try: 
            timeout = rclpy.duration.Duration(seconds=0.3)
            transform = self.tf_buffer.lookup_transform(target_frame, source_frame, stamp,timeout)
            translation_vec = np.array([transform.transform.translation.x,transform.transform.translation.y,transform.transform.translation.z])
            q = transform.transform.rotation

            r = Rotation.from_quat([q.x, q.y, q.z, q.w])
            points_rotated = r.apply(points_np)
            
            # Apply Translation
            points_map = points_rotated + translation_vec
            return points_map
        
        except Exception as e:
            self.get_logger().warn(f'Transform failed: {e}')
            return np.empty((0,3))

    def get_masks(self, colors, colors_box):
        'gets the colors of the points as imput and returns the color masks'
        # conversion of color spaces from rgb to oklab
        colors_rgb = colors.astype(np.float32) / 255
        colors_xyz = co.sRGB_to_XYZ(colors_rgb)
        colors_oklab = co.XYZ_to_Oklab(colors_xyz)

        colors_rgb_box = colors_box.astype(np.float32) / 255
        colors_xyz_box = co.sRGB_to_XYZ(colors_rgb_box)
        colors_oklab_box = co.XYZ_to_Oklab(colors_xyz_box)

        # assembling of color masks
        red_mask = (
            (self.thresh[0, 0] < colors_oklab[:, 1]) & (colors_oklab[:, 1] < self.thresh[0, 1]) & 
            (self.thresh[0, 2] < colors_oklab[:, 2]) & (colors_oklab[:, 2] < self.thresh[0, 3]) & 
            (self.thresh[0, 4] < colors_oklab[:, 0]) & (colors_oklab[:, 0] < self.thresh[0, 5])
        )
        green_mask = (
            (self.thresh[1, 0] < colors_oklab[:, 1]) & (colors_oklab[:, 1] < self.thresh[1, 1]) & 
            (self.thresh[1, 2] < colors_oklab[:, 2]) & (colors_oklab[:, 2] < self.thresh[1, 3]) & 
            (self.thresh[1, 4] < colors_oklab[:, 0]) & (colors_oklab[:, 0] < self.thresh[1, 5])
        )
        blue_mask = (
            (self.thresh[2, 0] < colors_oklab[:, 1]) & (colors_oklab[:, 1] < self.thresh[2, 1]) & 
            (self.thresh[2, 2] < colors_oklab[:, 2]) & (colors_oklab[:, 2] < self.thresh[2, 3]) & 
            (self.thresh[2, 4] < colors_oklab[:, 0]) & (colors_oklab[:, 0] < self.thresh[2, 5])
        )
        wood_mask = (
            (self.thresh[3, 0] < colors_oklab[:, 1]) & (colors_oklab[:, 1] < self.thresh[3, 1]) & 
            (self.thresh[3, 2] < colors_oklab[:, 2]) & (colors_oklab[:, 2] < self.thresh[3, 3]) & 
            (self.thresh[3, 4] < colors_oklab[:, 0]) & (colors_oklab[:, 0] < self.thresh[3, 5])
        )
        box_mask = (            
            (self.thresh[4, 0] < colors_oklab_box[:, 1]) & (colors_oklab_box[:, 1] < self.thresh[4, 1]) & 
            (self.thresh[4, 2] < colors_oklab_box[:, 2]) & (colors_oklab_box[:, 2] < self.thresh[4, 3]) & 
            (self.thresh[4, 4] < colors_oklab_box[:, 0]) & (colors_oklab_box[:, 0] < self.thresh[4, 5])
            )
        
        return red_mask, green_mask, blue_mask, wood_mask, box_mask
    
    def process_clusters(self, points_3d, box = False):
        """
        Input: points_3d (N, 3) numpy array of filtered XYZ coordinates
        Output: List of centroids [x, y, z] for valid objects
        """
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)

        if len(points_3d) < self.min_samples:  #TODO use thsi parameter as tuning and define it in the __init__
            return []

        # 1. Run Clustering (Very fast on <2000 points)
        # Returns labels like [0, 0, 1, -1, 0, 1...] (-1 is noise)
        labels = dbscan.fit_predict(points_3d)
        
        valid_centroids = []
        
        # Get unique labels (skip -1 which is noise)
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        for label in unique_labels:
            # 2. Extract Points for this specific cluster
            # Boolean indexing is fast
            cluster_mask = (labels == label)
            cluster_points = points_3d[cluster_mask]
            
            # 3. FAST Geometric Check (Axis-Aligned Bounding Box)
            # We calculate the dimensions of the cluster
            min_p = np.min(cluster_points, axis=0)
            max_p = np.max(cluster_points, axis=0)
            dims = max_p - min_p # [width_x, width_y, height_z]
            
            # Check 1: Is the size roughly correct?
            # You can get more specific (e.g., check X vs Y vs Z) if rotation is known
            if box == False:
                if not (self.obj_width - self.obj_tolerance < np.max(dims) < self.obj_width + self.obj_tolerance):
                    continue # Skip this cluster, it's too big/small
            if box: 
                if not (self.box_min_width < np.max(dims)< self.box_max_width):
                    # self.get_logger().info(f'object is not the size of a box')
                    continue # skip this cluster, its too big/small
                
            # Check 2: Density Check (Optional but recommended)
            # If it's the right size but has only 15 points, it might be a ghost reflection
            # A real solid object should have many points

            if len(cluster_points) < self.min_samples: 
                continue

            # 4. Calculate Centroid
            centroid = np.mean(cluster_points, axis=0)
            valid_centroids.append(centroid)

        return valid_centroids
    
    def is_close_to_obstacle(self, x, y):
        """
        Checks if a given (x, y) point is within a tunable distance of an obstacle.
        """
        if self.occupancy_grid is None:
            self.get_logger().warn("Occupancy grid is not yet available.")
            return False

        # 1. Fetch tuning parameters (assuming you declared these in __init__)
        # Distance to check around the point (in meters)
        search_radius_m = self.get_parameter("obstacle_distance_m").value
        # Value at which a cell is considered solid
        occ_threshold = self.get_parameter("occupancy_threshold").value

        # 2. Extract map metadata
        resolution = self.occupancy_grid.info.resolution
        width = self.occupancy_grid.info.width
        height = self.occupancy_grid.info.height
        origin_x = self.occupancy_grid.info.origin.position.x
        origin_y = self.occupancy_grid.info.origin.position.y

        # 3. Convert physical (x, y) to grid indices (col, row)
        center_col = int((x - origin_x) / resolution)
        center_row = int((y - origin_y) / resolution)

        # Quick check: is the point even inside the map?
        if not (0 <= center_col < width and 0 <= center_row < height):
            self.get_logger().warn("Point is outside the map bounds.")
            return True # Often safer to treat out-of-bounds as an obstacle

        # 4. Convert search radius from meters to cells
        radius_cells = math.ceil(search_radius_m / resolution)

        # 5. Search the bounding box around the target point
        for r_offset in range(-radius_cells, radius_cells + 1):
            for c_offset in range(-radius_cells, radius_cells + 1):
                
                # Check if the offset is within the circular radius (Euclidean distance)
                if math.sqrt(r_offset**2 + c_offset**2) <= radius_cells:
                    
                    check_row = center_row + r_offset
                    check_col = center_col + c_offset

                    # Ensure the cell we are checking is within grid bounds
                    if 0 <= check_row < height and 0 <= check_col < width:
                        
                        # Calculate the 1D index for the flat data array
                        # Index = row * width + col
                        index = check_row * width + check_col
                        cell_value = self.occupancy_grid.data[index]

                        # Check against the tunable threshold
                        if cell_value >= occ_threshold:
                            return True  # Found an obstacle!

        # If the loop finishes without triggering the threshold, the area is clear
        return False

    def publish_detection(self, centroid, header, color):
        # ToDo add
        msg = PointStamped()
        msg.header = header
        msg.point.x = centroid[0]
        msg.point.y = centroid[1]
        msg.point.z = centroid[2]

        if self.is_close_to_obstacle(centroid[0],centroid[1]):
            # self.get_logger().info(f'point x={centroid[0]}, y={centroid[1]} is too close to an object')
            return
        # else:
            # self.get_logger().info(f'Point (x,y){(centroid[0],centroid)} will now be published as an object')


        if color == 'red':
            self.red_centroid_pub.publish(msg)
        elif color == 'green':
            self.green_centroid_pub.publish(msg)
        elif color == 'blue':
            self.blue_centroid_pub.publish(msg)
        elif color == 'wood':
            self.wood_centroid_pub.publish(msg)
        elif color =='box':
            self.box_centroid_pub.publish(msg)

    def get_thresholds(self):

        comp_colors_rgb = np.array([
            [140, 45, 35], #red
            [0, 70, 57], #green
            [0, 83, 125], # blue
            [100, 75, 52], #wood
            [71, 93, 102] # grey box
            ])
                
        comp_colors_rgb = comp_colors_rgb / 255.0
        comp_colors_xyz = co.sRGB_to_XYZ(comp_colors_rgb)
        comp_colors_oklab = co.XYZ_to_Oklab(comp_colors_xyz)
        self.get_logger().info(f'comp_colors_oklab\n red: {comp_colors_oklab[0,:]} \n green: {comp_colors_oklab[1,:]}\n blue {comp_colors_oklab[2,:]}\n wood{comp_colors_oklab[3,:]}\n box{comp_colors_oklab[4,:]}')
        
        # define tolerances
        # loose thresholds tol_red = 0.04    tol_green = 0.02 tol_blue = 0.025 tol_wood = 0.012 tol_box = 0.02    
  
        # medium trehsholds
        tol_red = 0.03   
        tol_green = 0.015
        tol_blue = 0.02
        tol_wood = 0.011
        tol_box = 0.0002  

        # strict thresholds
        # tol_red = 0.02 tol_green = 0.01 tol_blue = 0.015 tol_wood = 0.01 tol_box = 0.02  

        thresh_red_L_low = 0.3 # these L thresholds are very very loose
        thresh_red_L_high = 0.55
        thresh_red_a_low = comp_colors_oklab[0,1] - tol_red
        thresh_red_a_high = comp_colors_oklab[0,1] + tol_red
        thresh_red_b_low = comp_colors_oklab[0,2] - tol_red
        thresh_red_b_high = comp_colors_oklab[0,2] + tol_red

        thresh_green_L_low = 0.25 # these L thresholds are very very loose
        thresh_green_L_high = 0.45
        thresh_green_a_low = comp_colors_oklab[1,1] - tol_green
        thresh_green_a_high = comp_colors_oklab[1,1] + tol_green
        thresh_green_b_low = comp_colors_oklab[1,2] - tol_green
        thresh_green_b_high = comp_colors_oklab[1,2] + tol_green

        thresh_blue_L_low = 0.3 # these L thresholds are very very loose
        thresh_blue_L_high = 0.55
        thresh_blue_a_low = comp_colors_oklab[2,1] - tol_blue
        thresh_blue_a_high = comp_colors_oklab[2,1] + tol_blue
        thresh_blue_b_low = comp_colors_oklab[2,2] - tol_blue
        thresh_blue_b_high = comp_colors_oklab[2,2] + tol_blue

        thresh_wood_L_low = 0.3 # these L thresholds are very very loose
        thresh_wood_L_high = 0.5
        thresh_wood_a_low = comp_colors_oklab[3,1] - tol_wood
        thresh_wood_a_high = comp_colors_oklab[3,1] + tol_wood
        thresh_wood_b_low = comp_colors_oklab[3,2] - tol_wood
        thresh_wood_b_high = comp_colors_oklab[3,2] + tol_wood

        thresh_box_L_low = 0.45
        thresh_box_L_high = 0.52
        thresh_box_a_low = comp_colors_oklab[4,1] - tol_box
        thresh_box_a_high = comp_colors_oklab[4,1] + tol_box
        thresh_box_b_low = comp_colors_oklab[4,2] - tol_box
        thresh_box_b_high = comp_colors_oklab[4,2] + tol_box
        

        thresh = np.array([[thresh_red_a_low,thresh_red_a_high,thresh_red_b_low, thresh_red_b_high, thresh_red_L_low, thresh_red_L_high],
                         [thresh_green_a_low,thresh_green_a_high,thresh_green_b_low, thresh_green_b_high, thresh_green_L_low, thresh_green_L_high],
                         [thresh_blue_a_low,thresh_blue_a_high,thresh_blue_b_low, thresh_blue_b_high, thresh_blue_L_low, thresh_blue_L_high],
                         [thresh_wood_a_low,thresh_wood_a_high,thresh_wood_b_low, thresh_wood_b_high, thresh_wood_L_low, thresh_wood_L_high],
                         [thresh_box_a_low,thresh_box_a_high,thresh_box_b_low, thresh_box_b_high, thresh_box_L_low, thresh_box_L_high]
                         ])
        
        return thresh


def main():
    rclpy.init()
    node = Detection()

    ex = MultiThreadedExecutor()
    ex.add_node(node)

    try:
        # rclpy.spin(node)
        ex.spin()
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()


if __name__ == '__main__':
    main()