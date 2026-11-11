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
pixi run ros2 bag record -o icp_data2 /lidar/scan /phidgets/imu/data_raw /phidgets/motor/encoders /realsense/depth/color/points /realsense/depth/image_rect_raw /realsense/color/image_raw/compressed

pixi run ros2 bag record -o arm_camera_detection /arm/camera/image_raw
```