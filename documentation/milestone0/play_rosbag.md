Play rosbag
```bash
pixi run ros2 bag play --read-ahead-queue-size 100 -r 1.0 --clock 100 --start-paused rosbags/milestone0_final_v2
```
Open Rviz
```bash
pixi run rviz2
```
We created an rviz config file with all the topics loaded
```bash
pixi run rviz2 -d src/all_enabled_config.rviz
```

Create static transforms for measurements reference frames (running a launch file)
```bash
pixi run ros2 launch robp_launch frames_launch.xml
```
Run odometry
```bash
pixi run ros2 run odometry odometry
```

Check tf tree
```bash
ros2 run tf2_tools view_frames
```


```bash
pixi run ros2 bag play --read-ahead-queue-size 100 -r 1.0 --clock 100 --start-paused rosbags/obj_det_odom_opti_rosbag
```



