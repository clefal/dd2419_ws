# Global Planner – Internal Documentation

## Purpose

The Global Planner computes a collision-free path from the current robot pose to a goal pose.

It operates entirely in the **map frame** and performs grid-based planning on a probabilistic occupancy grid.

---

## Inputs

### Subscribed Topics
- `/map/occupancy_grid` (OccupancyGrid)  
  Probabilistic map built from LiDAR.

- `/nav/goal` (PoseStamped)  
  Navigation goal in map frame (from goal manager).

- `/nav/objects/cubes` (PoseArray)  
  All known cube positions (used to overlay obstacles).

- `/nav/target/cube` (PoseStamped)  
  Current navigation target (excluded from obstacle overlay).

### TF
- `map → base_link`  
  Used to compute the start position.

---

## Processing Pipeline

1. Copy occupancy grid.
2. Inflate static obstacles:
   - Hard core (robot radius constraint)
   - Soft halo (penalized cost region)
3. Overlay cube obstacles (except active target).
4. Run **Weighted A\*** on 8-connected grid.
5. Convert grid path to world coordinates.
6. Set final pose orientation to goal yaw.

---

## Outputs

### Published Topics
- `/nav/global_path` (nav_msgs/Path)  
  Sequence of waypoints in map frame.  
  Final pose includes goal orientation.

- `/nav/planning_grid` (OccupancyGrid)  
  Inflated + cube-overlay grid for visualization/debugging.

---

## Key Parameters

- `occ_lethal` – Threshold for obstacle cells.
- `unknown_is_lethal` – Whether unknown space blocks planning.
- `w_heuristic` – Weighted A* heuristic scaling.
- `allow_diagonal` – Enable 8-connected planning.
- `occ_cost_scale` – Soft penalty scaling.

---

## Notes

- Planning is triggered on new goals.
- Operates fully in map frame.
- Ensures robot-radius safety via inflation.