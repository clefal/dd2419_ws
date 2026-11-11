TOPIC SUBSCRIBED 
/arm/camera/image_raw
Type: sensor_msgs/msg/Image
Purpose: Camera feed used for cube detection

/arm/action
Type: std_msgs/msg/String
Purpose: High-level commands to control the arm

Supported commands:
START
PICK_UP
DROP

TOPICS PUBLISHED  
/arm/control
Type: robp_interfaces/msg/ArmControl
Purpose: Sends joint positions and movement durations to the robotic arm

Fields:
position: list of 6 joint angles (0 gripper, 1 gripper rotation, 2-4 joints from gripper to base, 5 base rotation)
120 is middle, for 2 and 4 smaller is forward, for 3 smaller is backward 
time: list of durations for each joint movement (ms)

/arm/result
Type: std_msgs/msg/String
Purpose: Reports result of actions

Possible messages:
START_SUCCESS
START_FAIL: arm is not in start 
PICK_UP_SUCCESS
PICK_UP_FAIL_NO_OBJECT: No object after attempted pickup
PICK_UP_FAIL_RANGE: object not within range 
PICK_UP_FAIL_NO_IDLE: Did not attempt pickup since arm not in idle position
DROP_SUCCESS
DROP_FAIL_NO_OBJECT: not holding object 

ARM STATES 

start
Initial state when node starts
Moves arm into idle position

idle
Arm is stationary in default position

detect
Camera actively searching for cube
When detections stabalize either adjust arm position to cube or pickup cube 

pickup
Executes pickup sequence: Close gripper and Return to idle position 
If success transition into holding, if fail transition into idle

holding
Arm is holding an object

drop
arm is droping of an object 