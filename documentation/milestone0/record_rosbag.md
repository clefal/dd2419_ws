We want to record the following topics
- /lidar/scan
- /phidgets/imu
- /phidgets/motor/encoders
- /realsense/depth/color/points
- /realsense/depth/image_rect_raw
- /realsense/color/image_raw
- /arm/feedback
- /arm/camera/image_raw

So we open the rosbag folder
```bash
mkdir rosbags
```
and then run
```bash
pixi run ros2 bag record -o milestone0 /lidar/scan /phidgets/imu /phidgets/motor/encoders /realsense/depth/color/points /realsense/depth/image_rect_raw /realsense/color/image_raw/compressed /arm/feedback arm/camera/image_raw
```