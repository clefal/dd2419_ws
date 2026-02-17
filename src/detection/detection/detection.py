#!/usr/bin/env python
import numpy as np
import colour as co
import rclpy
import time
from rclpy.node import Node

from sklearn.cluster import DBSCAN

from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointField
from geometry_msgs.msg import PointStamped
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs # Required for transform_points
from scipy.spatial.transform import Rotation
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

import ctypes
import struct


class Detection(Node):

    def __init__(self):
        super().__init__('detection')

        # Initialize the publisher
        self._pub = self.create_publisher(
            PointCloud2, '/camera/depth/color/ds_points', 10, callback_group=ReentrantCallbackGroup())
        
        self.red_centroid_pub = self.create_publisher(PointStamped, '/detection/objects/red_cube', 10, callback_group=ReentrantCallbackGroup())
        self.green_centroid_pub = self.create_publisher(PointStamped, '/detection/objects/green_cube', 10, callback_group=ReentrantCallbackGroup())
        self.blue_centroid_pub = self.create_publisher(PointStamped, '/detection/objects/blue_cube', 10, callback_group=ReentrantCallbackGroup())
        self.wood_centroid_pub = self.create_publisher(PointStamped, '/detection/objects/wood_cube', 10, callback_group=ReentrantCallbackGroup())

        # Subscribe to point cloud topic and call callback function on each received message
        self.create_subscription(
            PointCloud2, '/realsense/depth/color/points', self.cloud_callback, 10, callback_group=ReentrantCallbackGroup())
        
        self.thresh = self.get_thresholds()

        # initialize clustering parameters
        self.min_samples = 5 # min number of samples to be considered one object
        self.eps = 0.02 # ponints within this distance to each other are considered one object
        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.obj_width = 0.03  # meters
        self.obj_width = 0.03 # meters
        self.tolerance = 0.03    # +/- 3cm tolerance

        # initialize TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # initialize point buffering
        self.point_buffers = {'red': [], 'green':[], 'blue': [], 'wood':[]}
        self.buffer_size = 3 # number of pointclouds we buffer before performing the clustering



    def cloud_callback(self, msg: PointCloud2):
        """
        Takes point cloud readings to detect objects.
        This function is called for every message that is published on the '/camera/depth/color/points' topic.
        """

        # TODO for the future, if it becomes a bottleneck: merge the messages into onemessage that is published
        # this is for sure cleaner since we currently have to handle multiple messages at the same time if we detect multiple things at the same time
        gen = pc2.read_points_numpy(msg, skip_nans=True)
        points = gen[:, :3]
        rgb_uint32 = gen[:, 3].view(np.uint32)
        colors = np.empty((len(rgb_uint32), 3), dtype=np.uint8)
        colors[:, 0] = (rgb_uint32 >> 16) & 255
        colors[:, 1] = (rgb_uint32 >> 8) & 255
        colors[:, 2] = rgb_uint32 & 255

        # geometrical filter
        max_dist = 2
        max_height = 0.05   
        min_height = 0.08
        geom_mask = ((points[:,2] < max_dist) & (points[:,1] > max_height) & (points[:,1] < min_height))
        # the cleanest solution is to filter the points in the odom/map frame this should be implemented in the future
        # also it should be checked if the 
        # TODO filter out the floor as well!

        points_f = points[geom_mask]
        colors_f = colors[geom_mask]

        points_map = self.transform_points_to_map(points_f, msg.header)


        red_mask, green_mask, blue_mask, wood_mask = self.get_masks(colors_f) # returns the color masks based on threshold values

        # Chek how many red,green,... points we have
        red_counter = np.sum(red_mask)
        green_counter = np.sum(green_mask)
        blue_counter = np.sum(blue_mask)
        wood_counter = np.sum(wood_mask)

        general_counter = red_counter + green_counter + blue_counter + wood_counter

        if general_counter == 0: return # end callback if we have no hits in general

        
        fields = [ # only for visualization in rviz, is actually not relevant
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            ]
        
        centroid_header = Header()
        centroid_header.stamp = msg.header.stamp  #this is a bit sus, since we are buffering the points
        centroid_header.frame_id = 'map'

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
                self.get_logger().info(f'red: {len(all_red_points)}')

                # only for visualization in rviz
                msg_red = pc2.create_cloud(centroid_header, fields, all_red_points)
                self._pub.publish(msg_red)

                for centroid in red_centroids: 
                    self.publish_detection(centroid, centroid_header, 'red')

                self.point_buffers['red'] = [] # after publishing clear the buffer
            
            # manage green_points
            if green_counter > 0: # add points to buffer if we have more than a minimum amount of hits
                green_points = points_map[green_mask]
                self.point_buffers['green'].append(green_points)

            if green_counter == 0 and len(self.point_buffers['green'])!= 0 : # clear the buffer if we dont see points in consecutive scans
                self.point_buffers['green'] = [] 

            if len(self.point_buffers['green'])>=self.buffer_size:
                all_green_points = np.vstack(self.point_buffers['green'])
                green_centroids = self.process_clusters(all_green_points)
                self.get_logger().info(f'green: {len(all_green_points)}')

                # only for visualization in rviz
                msg_green = pc2.create_cloud(centroid_header, fields, all_green_points)
                self._pub.publish(msg_green)

                for centroid in green_centroids: 
                    self.publish_detection(centroid, centroid_header, 'green')
                    
                self.point_buffers['green'] = [] # after publishing clear the buffer

            # manage blue_points
            if blue_counter > 0: # add points to buffer if we have more than a minimum amount of hits
                blue_points = points_map[blue_mask]
                self.point_buffers['blue'].append(blue_points)

            if blue_counter == 0 and len(self.point_buffers['blue'])!= 0 : # clear the buffer if we dont see points in consecutive scans
                self.point_buffers['blue'] = [] 

            if len(self.point_buffers['blue'])>=self.buffer_size:
                all_blue_points = np.vstack(self.point_buffers['blue'])
                blue_centroids = self.process_clusters(all_blue_points)
                self.get_logger().info(f'blue: {len(all_blue_points)}')

                # only for visualization in rviz
                msg_blue = pc2.create_cloud(centroid_header, fields, all_blue_points)
                self._pub.publish(msg_blue)

                for centroid in blue_centroids: 
                    self.publish_detection(centroid, centroid_header, 'blue')
                    
                self.point_buffers['blue'] = [] # after publishing clear the buffer

            # manage wood_points
            if wood_counter > 0: # add points to buffer if we have more than a minimum amount of hits
                wood_points = points_map[wood_mask]
                self.point_buffers['wood'].append(wood_points)

            if wood_counter == 0 and len(self.point_buffers['wood'])!= 0 : # clear the buffer if we dont see points in consecutive scans
                self.point_buffers['wood'] = [] 

            if len(self.point_buffers['wood'])>=self.buffer_size:
                all_wood_points = np.vstack(self.point_buffers['wood'])
                wood_centroids = self.process_clusters(all_wood_points)
                self.get_logger().info(f'wood: {len(all_wood_points)}')

                # only for visualization in rviz
                msg_wood = pc2.create_cloud(centroid_header, fields, all_wood_points)
                self._pub.publish(msg_wood)

                # for centroid in wood_centroids:     # so that marius can experiment with it i will uncomment this line 
                    #self.publish_detection(centroid, centroid_header, 'wood')
                    
                self.point_buffers['wood'] = [] # after publishing clear the buffer
            

    def transform_points_to_map(self, points_np, header :Header):
        """
        Converts an (N, 3) numpy array of points from camera frame to map frame.
        """
        source_frame = header.frame_id
        target_frame = 'map'
        stamp = header.stamp
        if len(points_np) == 0:
            self.get_logger().warn(f'transform_points_to_map() had an empty point array as input')
            return np.empty(0,3)
        

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            ]
            
        
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



    def get_masks(self, colors):
        'gets the colors of the points as imput and returns the color masks'
        # conversion of color spaces from rgb to oklab
        colors_rgb = colors.astype(np.float32) / 255
        colors_xyz = co.sRGB_to_XYZ(colors_rgb)
        colors_oklab = co.XYZ_to_Oklab(colors_xyz)

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

        return red_mask, green_mask, blue_mask, wood_mask
    


    def process_clusters(self, points_3d):
        """
        Input: points_3d (N, 3) numpy array of filtered XYZ coordinates
        Output: List of centroids [x, y, z] for valid objects
        """
        if len(points_3d) < self.min_samples:  #TODO use thsi parameter as tuning and define it in the __init__
            return []

        # 1. Run Clustering (Very fast on <2000 points)
        # Returns labels like [0, 0, 1, -1, 0, 1...] (-1 is noise)
        labels = self.dbscan.fit_predict(points_3d)
        
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
            if not (self.obj_width - self.tolerance < np.max(dims) < self.obj_width + self.tolerance):
                continue # Skip this cluster, it's too big/small
                
            # Check 2: Density Check (Optional but recommended)
            # If it's the right size but has only 15 points, it might be a ghost reflection
            # A real solid object should have many points

            if len(cluster_points) < self.min_samples: 
                continue

            # 4. Calculate Centroid
            centroid = np.mean(cluster_points, axis=0)
            valid_centroids.append(centroid)

        return valid_centroids
    

    def publish_detection(self, centroid, header, color):
        # ToDo add
        msg = PointStamped()
        msg.header = header
        msg.point.x = centroid[0]
        msg.point.y = centroid[1]
        msg.point.z = centroid[2]

        if color == 'red':
            self.red_centroid_pub.publish(msg)
        elif color == 'green':
            self.green_centroid_pub.publish(msg)
        elif color == 'blue':
            self.blue_centroid_pub.publish(msg)
        elif color == 'wood':
            self.wood_centroid_pub.publish(msg)


    def get_thresholds(self):

        comp_colors_rgb = np.array([
            [140, 45, 35], #red
            [0, 70, 57], #green
            [0, 83, 125], # blue
            [100, 75, 52] #wood
            ])
                
        comp_colors_rgb = comp_colors_rgb / 255.0
        comp_colors_xyz = co.sRGB_to_XYZ(comp_colors_rgb)
        comp_colors_oklab = co.XYZ_to_Oklab(comp_colors_xyz)
        self.get_logger().info(f'comp_colors_oklab\n red: {comp_colors_oklab[0,:]} \n green: {comp_colors_oklab[1,:]}\n blue {comp_colors_oklab[2,:]}\n wood{comp_colors_oklab[3,:]}')
        
        # define tolerances
        tol_red = 0.02
        tol_green = 0.01
        tol_blue = 0.015
        tol_wood = 0.01

        # thresh_red_L_low = comp_colors_oklab[0,0] - 0.15
        # thresh_red_L_high = comp_colors_oklab[0,0] + 0.15
        thresh_red_L_low = 0.0
        thresh_red_L_high = 1.0
        thresh_red_a_low = comp_colors_oklab[0,1] - tol_red
        thresh_red_a_high = comp_colors_oklab[0,1] + tol_red
        thresh_red_b_low = comp_colors_oklab[0,2] - tol_red
        thresh_red_b_high = comp_colors_oklab[0,2] + tol_red

        # thresh_green_L_low = comp_colors_oklab[1,0] - 0.25
        # thresh_green_L_high = comp_colors_oklab[1,0] + 0.25
        thresh_green_L_low = 0.0
        thresh_green_L_high = 1.0
        thresh_green_a_low = comp_colors_oklab[1,1] - tol_green
        thresh_green_a_high = comp_colors_oklab[1,1] + tol_green
        thresh_green_b_low = comp_colors_oklab[1,2] - tol_green
        thresh_green_b_high = comp_colors_oklab[1,2] + tol_green

        # thresh_blue_L_low = comp_colors_oklab[2,0] - 0.15
        # thresh_blue_L_high = comp_colors_oklab[2,0] + 0.15
        thresh_blue_L_low = 0.0
        thresh_blue_L_high = 1.0
        thresh_blue_a_low = comp_colors_oklab[2,1] - tol_blue
        thresh_blue_a_high = comp_colors_oklab[2,1] + tol_blue
        thresh_blue_b_low = comp_colors_oklab[2,2] - tol_blue
        thresh_blue_b_high = comp_colors_oklab[2,2] + tol_blue

        # thresh_wood_L_low = comp_colors_oklab[3,0] - 0.02
        # thresh_wood_L_high = comp_colors_oklab[3,0] + 0.02
        thresh_wood_L_low = 0.0
        thresh_wood_L_high = 1.0
        thresh_wood_a_low = comp_colors_oklab[3,1] - tol_wood
        thresh_wood_a_high = comp_colors_oklab[3,1] + tol_wood
        thresh_wood_b_low = comp_colors_oklab[3,2] - tol_wood
        thresh_wood_b_high = comp_colors_oklab[3,2] + tol_wood

        thresh = np.array([[thresh_red_a_low,thresh_red_a_high,thresh_red_b_low, thresh_red_b_high, thresh_red_L_low, thresh_red_L_high],
                         [thresh_green_a_low,thresh_green_a_high,thresh_green_b_low, thresh_green_b_high, thresh_green_L_low, thresh_green_L_high],
                         [thresh_blue_a_low,thresh_blue_a_high,thresh_blue_b_low, thresh_blue_b_high, thresh_blue_L_low, thresh_blue_L_high],
                         [thresh_wood_a_low,thresh_wood_a_high,thresh_wood_b_low, thresh_wood_b_high, thresh_wood_L_low, thresh_wood_L_high]])
        
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