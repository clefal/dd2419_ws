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
pixi run ros2 bag record -o obj_detection /phidgets/motor/encoders /realsense/depth/color/points /realsense/depth/image_rect_raw /realsense/color/image_raw/compressed /realsense/color/image_raw
```

__Some words about the Code:__


__Color Thresholding and Geometric Filtering__:

- The core concept is a simple thresholding for the 3d-Color-Points
- First we threshold for distance and height and already apply that geometric filter to the points
    - this is currently done in the frame of the 3d point cloud
    - one could also do that in the map frame, that would be a bit easier (but it makes distance thresholding a lot more difficult and a lot more expensive as we need transforms of the whole pointcloud) to understand would require tf lookups though
- Afterwards we threshold the color using oklab color space
    - oklab color space is an optimized color space (similar to rgb or hsv) but optimized for purposes like color detection using cameras
    - it has 3 Components
        - L: perceptual lightness ranging from 0% to 100%
        - a: for green and red values (green: -0.5, red: 0.5)
        - b: for blue and yellow values (blue: -0.5, yellow: 0.5)
    - This color space does better at handling differences in illumniation 
        - when thresholding in rgb space the a color A that is close to another color B in terms of their coordinates (e.G. (150, 10, 10) and (140, 10, 20)) are not actually close to each other in terms of how they look in different lightings, OKlab is supposed to handle this better
        - https://en.wikipedia.org/wiki/Oklab_color_space
    - As of right now we only threshold the Components a and b but that can and probably must be changed for future detection algorithms
    - This detection currently outperforms the rgb detection I used in the bootcamp

- rgb values of object, measured with the realsense (2 measurements at 2 different distances)
    - red: 137 55 50  / 145 37 17
    - green: 0 72  58 / 1 67 56 
    - blue: 1 90 134 / 2 76 117
    - wood: 111 77 49 / 90 73 54
    --> theese values are roughly averaged and are used to calculate oklab threshold values 

__Buffering and Clustering:__

- after thresholding we first buffer the points
    - if N consecutive Scans have points of one specific color in them all these points are converted to map frame and saved in the buffer
- once we have N consecutive Scans the clustering starts picking up  its work
    - currently we use DBSCAN to get annotations of the points if it is part of an onject (and wich object) it is part of
    - afterwards we check the dimensions of the objects if it is within the tolerance we will accept the object and calculate the centroid of it
    - then we return the centroid and publish it
    - we do that for all objects, so we do the same thing for every cube
        --> currently we publish all centroids that are detected, we should implement something that checks wich one is more likely since probably one will be noise (if 2 blue objects are detected than one is probablyl noise)



__What's next?__

- we still need to filter out stuff that is not in the workspace
- we should do some more parameter tuning of the clustering and the color thresholding (hard to estimate when we start overfitting to the rosbags we have)
- we need to integrate the detection into the system so that the messages we poblished are actually used by e.G. the navigator
- we have to implement the box detection (currently only doing cubes)
    - that will lead to having more points to deal with and therefore needs a different size filter when clustering




  
