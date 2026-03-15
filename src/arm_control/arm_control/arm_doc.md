Messages published by arm to arm/result with type String
”START_SUCCESS”: arm is in start position
"PICK_UP_SUCCESS”: arm is holding object
"PICK_UP_FAIL_NO_OBJECT”: object is not in arm after attempted pick_up
"PICK_UP_FAIL_NO_START”: did not attempt pick_up because arm was not in start position
”DROP_SUCCESS”: drop successfull, arm is in start position
”DROP_FAIL_NO_OBJECT”: did not attempt drop because robot was not holding an object

Messages published by goal manager to arm/status with type string 
”START”: Go into start position
”PICK_UP”: Pick up object
”DROP”: Drop object 

POSITIONS:
idle: position used in between pick ups checks from this position of object is reachable 
initial_pickup: middle of pick up range and just above ground when arm closes 
