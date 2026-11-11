# Goal Manager – Internal Documentation

## Purpose

The Goal Manager is the high-level task coordinator of the robot.

It connects:
- Navigation (global planner + controller)
- occupancy grid
- Object detection
- Manipulation (robotic arm)

It manages *what* the robot should do next (task logic), not *how* it moves or plans paths.

All logic operates in the **map frame**.

---

## State Machine Overview

The system is implemented as a finite state machine:

IDLE
↓
INITIALIZATION
↓
SEARCH
↓
APPROACH_OBJECT
↓
WAIT_PICKUP_RESULT
↓
RETURN_HOME
↓
WAIT_DROP_RESULT
↓
(next cube or SEARCH)



## State Descriptions

### IDLE
Robot is inactive. Used as a safe resting state.

---

### INITIALIZATION
- Sends `START` command to the arm.
- Transitions immediately to `SEARCH`.

---

### SEARCH
- Robot navigates to a predefined search position (hardcoded right now, replace with algorithm later on).
- Cube detections are accepted.
- If cubes are known (from mapfile or detection), the closest cube is selected.
- Transitions to `APPROACH_OBJECT`.

---

### APPROACH_OBJECT
- Publishes a navigation goal to a standoff position in front of the selected cube.
- When `/nav/status` reports `REACHED`:
  - Sends `PICK_UP` to the arm.
  - Transitions to `WAIT_PICKUP_RESULT`.

---

### WAIT_PICKUP_RESULT
- Waits for `/arm/result`.

If `PICK_UP_SUCCESS` (later triggered via arm camera):
- Removes the picked cube from the internal cube list.
- Publishes updated cube topics.
- Sends navigation goal to the drop-off box.
- Transitions to `RETURN_HOME`.

If pickup fails:
- Logs warning.
- Remains available for further actions.

---

### RETURN_HOME
- Robot navigates to the drop-off box.
- Cube detections are still accepted (for obstacle avoidance).
- Goal switching is disabled while carrying a cube.

When `/nav/status` reports `REACHED`:
- Sends `DROP` to the arm.
- Transitions to `WAIT_DROP_RESULT`.

---

### WAIT_DROP_RESULT
- Waits for `/arm/result`.

If `DROP_SUCCESS`:
- If cubes remain → select nearest cube and transition to `APPROACH_OBJECT`.
- Otherwise → transition back to `SEARCH`.

---

## Cube Handling Logic

### Static Cubes (Mapfile)
- Loaded from TF frames (`object0`, `object1`, ...).
- Stored as `(x, y)` in map frame.
- Immediately usable as navigation targets.

### Detected Cubes
- Subscribed from `/detection/objects/blue_cube`. (later generalized to all cubes, once detection works flawlessly)
- Transformed into map frame.
- Deduplicated using a merge radius (10 cm).
- Added to internal cube list.

During `SEARCH` and `APPROACH_OBJECT`, detections may trigger goal switching.  
During `RETURN_HOME`, detections update the world model but do not change the current target.

---

## Target Selection Strategy

- Always selects the **closest cube to the robot**.
- Publishes:
  - `/nav/objects/cubes` (PoseArray of all known cubes)
  - `/nav/target/cube` (PoseStamped of active target)

Replanning is triggered by publishing a new navigation goal.

---

## Interfaces

### Subscribed Topics
- `/nav/status` – Navigation feedback (`REACHED`, `FAILED`)
- `/arm/result` – Arm pickup/drop results
- `/detection/objects/blue_cube` – Cube detections

### Published Topics
- `/nav/goal` – Navigation goal (PoseStamped)
- `/nav/objects/cubes` – All known cubes
- `/nav/target/cube` – Current target cube
- `/arm/action` – Arm commands (`START`, `PICK_UP`, `DROP`)

---

## Frames

- All planning goals are expressed in `map`.
- Robot pose obtained via TF (`map → base_link`).
- Workspace objects and drop-off box are provided as static TF frames.