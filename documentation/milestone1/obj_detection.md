**Readme about the detection part of the detection part of Milestone 1:**

__For playing back the rosbag run the following nodes /launch files:__
```bash
pixi run rviz2
pixi run ros2 run odometry odometry --ros-args -p use_sim_time:=true 
pixi run ros2 launch robp_launch frames_launch.xml
pixi run ros2 bag play --read-ahead-queue-size 100 -l -r 1.0 --clock 100 --start-paused ~/dd2419_ws/rosbags/obj_det_odom_opti_rosbag
pixi run ros2 run detection detection --ros-args -p use_sim_time:=true
pixi run mapping
```

__ssh into the Robot in Clemens Hotspot:__
```bash
sshpass -p 'group3' ssh group3@10.28.211.242
```

__Record rosbag:__
```bash
pixi run ros2 bag record -o obj_detection /phidgets/motor/encoders /realsense/depth/color/points /realsense/color/image_raw/compressed /phidgets/imu/data_raw /nav/is_turning /lidar/scan
```

__Some words about the Code:__


__Color Thresholding and Geometric Filtering__:

- The core concept is a simple thresholding for the 3d-Color-Points
- First we threshold for distance and height and already apply that geometric filter to the points
    - this is currently done in the frame of the 3d point cloud
    - one could also do that in the map frame, that would be a bit easier (but it makes distance thresholding a lot more difficult and a lot more expensive as we need transforms of the whole pointcloud) to understand, but would require tf lookups though
- Afterwards we threshold the color using oklab color space
    - oklab color space is an optimized color space (similar to rgb or hsv) but optimized for purposes like color detection using cameras
    - it has 3 Components
        - L: perceptual lightness ranging from 0% to 100%
        - a: for green and red values (green: -0.5, red: 0.5)
        - b: for blue and yellow values (blue: -0.5, yellow: 0.5)
    - This color space does better at handling differences in illumniation 
        - when thresholding in rgb space the a color A that is close to another color B in terms of their coordinates (e.G. (150, 10, 10) and (140, 10, 20)) are not actually close to each other in terms of how they look in different lightings, OKlab is supposed to handle this better
        - https://en.wikipedia.org/wiki/Oklab_color_space
    - As of right now we only threshold the components a and b but that can and probably must be changed for future detection algorithms

- rgb values of object, measured with the realsense (2 measurements at 2 different distances)
    - red: 137 55 50  / 145 37 17
    - green: 0 72  58 / 1 67 56 
    - blue: 1 90 134 / 2 76 117
    - wood: 111 77 49 / 90 73 54
    - box: 71 93 102
    --> theese values are roughly averaged and are used to calculate oklab threshold values 

__Buffering and Clustering:__

- after thresholding we first buffer the points
    - if N consecutive Scans have points of one specific color in them all these points are converted to map frame and saved in the buffer
- once we have N consecutive Scans the clustering starts picking up its work
    - currently we use DBSCAN to get annotations of the points if it is part of an onject (and wich object) it is part of
    - afterwards we check the width of the objects. If it is within the tolerance, we will accept the object and calculate the centroid of it
    - then we return the centroid and publish it
    - we do that for all objects, so we do the same thing for every cube


__Occupancy Check:__

- before publishing a point we check if it is in proximity to an occupied voxel
    - if it is, we will not publish it as it is a false positive
- This check only works if an Occupancy grid is available. The Subscription ot the occupancy grid is latched (so we always have the last sent message on the occupancy grid topic available) 
    - Even if no Occupancy grid is available the detection works but does not do the occupancy check
        - If this leads to false positives (e.g. because in the very beginning we have no Occupancy grid) then we could also disallow publishing of points when no map is available

- The Occupancy check is very powerful and even makes the buffering obsolete, it is probably more robust to still use the buffering but i tried it on different rosbags and it did not make a difference if we use buffer_size = 1 or buffer_size = 5

__Box Detection:__

- the box detection is kept somewhat seperate from the cube detection as it needs different geometric thresholding
- since the box is higher than the cubes we need to set the max_height used
    - if we would use the same max_height for the cubes as well then we would get many more points and therefore also a higher risk of false positives
    - since we only are looking for the color grey in the "box_points" we dont get as many false positives on the cardboard walls as they are wood colored
- the clustering is also a bit different, for cubes vs. box as teh box is bigger
- After clustering the "center" of the object is published as the average of all the points that we see of that object. 
    - so when facing the wide side of the box, the centroid is on the middle of the surface of that seen side 
    - So "center" is not acually the middle point of the box
    - the distance between the published "center" and the actual middle point of the box is also not constant but dependent on in what angle we look at the box
    - We probably have to deal with this problem in the future somehow

__What's next?__

- Test Test Test
    - thresholds are probably a bit light dependent so performance can vary with sun/light in the room
    - Observe how the "center" vs actual center problem makes tasks more difficult


__Random Information__

- the Pointcloud that we currently publish is only used for debugging
- If we need to clean up the code this could be the first thing that has to go...



  
