#!/usr/bin/env python
import rclpy 
import math
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, TransformStamped
from tf_transformations import quaternion_from_euler, euler_from_quaternion
from tf2_ros import Buffer, TransformListener, TransformBroadcaster, StaticTransformBroadcaster
from robp_interfaces.srv import GoalsAvailable, GetClosestCube, SetStatus, GetAllObjects, GetClosestBox, GetPosOfObj
from robp_interfaces.msg import ObjPose


## RENAME THIS NODE TO OBJECT MANAGER!!###
class Obj: 
    def __init__(self, id, x, y,  yaw=0 ,status = 'available', type='cube'):
        self.id = id
        self.first_x = x
        self.first_y = y
        self.last_x = x
        self.last_y = y
        self.first_yaw = yaw
        self.last_yaw = yaw
        self.status = status
        self.type = type

    def copy(self):
        return Obj(self.id, self.first_x, self.first_y, self.first_yaw, self.status, self.type)


        # things that could be changed in the future: 
        # add the stamp of the latest measurement
        # add detection counter for each object
        

class ObjectManager(Node):

    def __init__(self):
        super().__init__('object_manager')
        self.get_logger().info('Detection Manager node started.')

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._static_tf_broadcaster = StaticTransformBroadcaster(self)

        # subscribers

        self.sub_red_cube = self.create_subscription(PointStamped,'/detection/objects/red_cube', self.red_callback, 10)
        self.sub_green_cube = self.create_subscription(PointStamped,'/detection/objects/green_cube', self.green_callback, 10)
        self.sub_blue_cube = self.create_subscription(PointStamped,'/detection/objects/blue_cube', self.blue_callback, 10)
        self.sub_wood_cube = self.create_subscription(PointStamped,'/detection/objects/wood_cube', self.wood_callback, 10)
        self.sub_box = self.create_subscription(PointStamped,'/detection/objects/box', self.box_callback, 10)

        self.object_list = list()

        # services 
        self.srv_goals_available = self.create_service(GoalsAvailable,'object_manager/goals_available', self.goals_available_callback)
        self.srv_get_closest_cube = self.create_service(GetClosestCube,'object_manager/get_closest_cube', self.get_closest_cube_callback)
        self.srv_get_closest_box = self.create_service(GetClosestBox,'object_manager/get_closest_box', self.get_closest_box_callback)
        self.srv_set_status = self.create_service(SetStatus,'object_manager/set_status', self.set_status_callback)
        self.srv_get_all_objects = self.create_service(GetAllObjects,'object_manager/get_all_objects', self.get_all_objects_callback)
        self.srv_get_pos_of_obj = self.create_service(GetPosOfObj,'object_manager/get_pos_of_obj', self.get_pos_of_obj_callback)

        # load objects from the workspace file into the list
        self._fixed_frame = 'map'
        self._object_frame_prefix = 'object'
        self._box_frame = 'box'
        self._max_static_objects = 50

        self._static_loaded = False
        self.create_timer(0.5, self.get_points_from_csv_once)

        self.similarity_threshold = 0.2 # distance of detections that are combined into one object

# ----------------------------------

    def lookup_xy_yaw(self, parent_frame: str, child_frame: str):
        try:
            t = self._tf_buffer.lookup_transform(parent_frame, child_frame, rclpy.time.Time())
        except Exception:
            return None
        x = t.transform.translation.x
        y = t.transform.translation.y
        q = t.transform.rotation
        yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        return (x, y, yaw)

# --------------------------------

    def get_new_obj_idx(self):
        '''returns new unique object index'''
        if len(self.object_list) == 0:
            return 0
        else:
            return self.object_list[-1].id + 1
    
# ---------------------------------

    def check_similarity(self, obj :Obj):
        '''checks similarity of the obj with the objects in the object_list, returns the number of similar objects.
        If object is similar to another object then this object will be updated.'''
        similarity_counter = 0

        # maybe this can be done quicker with pandas or something like that, so if it becomes a problem then i can look into that again
        if len(self.object_list)>0:
            for idx, o in enumerate(self.object_list):
                if o.type == obj.type or o.type == 'map_cube':
                    # since we dont know the colors of the cubes from the map file we only do position comparison to check for similar objects
                    if abs(o.first_x - obj.first_x) < self.similarity_threshold and abs(o.first_y - obj.first_y) < self.similarity_threshold:
                        # if the object is similar (=close to another object and of same type)
                        updated_obj = o.copy()
                        updated_obj.last_x = obj.last_x
                        updated_obj.last_y = obj.last_y
                        updated_obj.last_yaw = obj.last_yaw
                        self.object_list[idx] = updated_obj

                        similarity_counter += 1
            
        return similarity_counter       
        
# ---------------------------------

    def publish_objects(self):
        '''publishes all objects from the object_list'''
        self.get_logger().info(f'Publishing {len(self.object_list)} objects')
        parent_frame = self._fixed_frame
        for obj in self.object_list:
            frame_name = f'{self._object_frame_prefix}{obj.id}'
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()    # maybe change this and actually take the timestamp from when the objects were published for that we need to save the stamp in the object list
            t.header.frame_id = parent_frame
            t.child_frame_id = frame_name

            t.transform.translation.x = obj.last_x
            t.transform.translation.y = obj.last_y
            t.transform.translation.z = 0.0
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0

            self._static_tf_broadcaster.sendTransform(t)
            # this now always broadcasts a static transform of the first detection of the point, the static transform somehow does not update itself
            # since we are not planning on using these tfs anyways but it was just a requirement from the ms2 i think it is okay
            # we could switch to dynamic broadcasts though, but i fear that if we wanted to use these transforms we could get some interpolation into the future errors since we
            # need to know the transform after we detected the object and not inbetween 2 detections
             
# ---------------------------------

    def get_points_from_csv_once(self):
        if self._static_loaded:
            return

        # get box position
        box_pose = self.lookup_xy_yaw(self._fixed_frame, self._box_frame)
        if box_pose is None:
            # workspace_loader not ready yet
            return
        bx, by, byaw = box_pose

        static_box_idx = self.get_new_obj_idx()
        static_box_obj = Obj(static_box_idx, bx, by, byaw, status='available', type ='box')

        if self.check_similarity(static_box_obj) == 0: # that means that 0 objects are similar to the static_box_object
            self.object_list.append(static_box_obj)

        
        # get object poritions from map file
        seeded = 0
        for i in range(self._max_static_objects):
            child = f'{self._object_frame_prefix}{i}'
            obj_pose = self.lookup_xy_yaw(self._fixed_frame, child)
            if obj_pose is None:
                # assume contiguous indices for the static transform idices; stop at first missing
                break
            ox, oy, _ = obj_pose

            static_cube_idx = self.get_new_obj_idx()
            static_cube_obj = Obj(static_cube_idx, ox, oy, yaw = 0, status='available', type = 'map_cube')

            if self.check_similarity(static_cube_obj) == 0: 
                self.object_list.append(static_cube_obj)
            
            seeded += 1

        if seeded > 0:
            self.get_logger().info(f'Seeded {seeded} cubes from static TF frames ({self._object_frame_prefix}0..).')
            # self.publish_topics()
        
        # find a way how to publish the objects in a function
        self.publish_objects()        
        self._static_loaded = True

# -------------------------

    def process_object(self,x , y, yaw, obj_type):
        idx = self.get_new_obj_idx()
        status = 'available'
        obj = Obj(idx, x, y, yaw, status=status, type=obj_type)

        if self.check_similarity(obj) == 0:  # if object is similar to zero objects then add it to the list
            self.object_list.append(obj)
        
# -------------------------
        
############ Object-Topic- Callbacks #############

# -------------------------
    def red_callback(self, msg  : PointStamped):
        yaw = 0
        obj_type = 'red_cube'
        self.process_object(msg.point.x, msg.point.y, yaw, obj_type)
        self.publish_objects()

    def green_callback(self, msg  : PointStamped):
        yaw = 0
        obj_type = 'green_cube'
        self.process_object(msg.point.x, msg.point.y, yaw, obj_type)
        self.publish_objects()

    def blue_callback(self, msg  : PointStamped):
        yaw = 0
        obj_type = 'blue_cube'
        self.process_object(msg.point.x, msg.point.y, yaw, obj_type)
        self.publish_objects()

    def wood_callback(self, msg  : PointStamped):
        yaw = 0
        obj_type = 'wood_cube'
        self.process_object(msg.point.x, msg.point.y, yaw, obj_type)
        self.publish_objects()

    def box_callback(self, msg  : PointStamped):
        yaw = 0
        obj_type = 'box'
        self.process_object(msg.point.x, msg.point.y, yaw, obj_type)
        self.publish_objects()        

# ------------------------

############ Service Callbacks #############

# -----------------------
    def goals_available_callback(self, req, res):
        
        res.goals_available = False

        for obj in self.object_list:
            if obj.status == 'available' and obj.type != 'box':
                res.goals_available = True # the variable name in the res object has to match the one defined in the goals_available.srv (see robp_interfaces)
                return res
            
        return res
        
        
# -----------------------
    def get_closest_cube_callback(self,req, res):
        self.get_logger().info(f'get_closest_CUBE_callback entered')
        closest_obj_id = None
        closest_obj_x = 0.0
        closest_obj_y = 0.0
        closest_obj_yaw = 0.0
        for obj in self.object_list:
            if obj.status == 'available' and obj.type != 'box':
                if closest_obj_id == None:
                    closest_obj_id = obj.id
                    closest_obj_x = obj.last_x
                    closest_obj_y = obj.last_y
                    closest_obj_yaw = obj.last_yaw
                    closest_distance = math.hypot(obj.last_x - req.robot_x, obj.last_y - req.robot_y)
                if math.hypot(obj.last_x - req.robot_x, obj.last_y - req.robot_y) < closest_distance:
                    closest_obj_id = obj.id
                    closest_obj_x = obj.last_x
                    closest_obj_y = obj.last_y
                    closest_obj_yaw = obj.last_yaw

        if closest_obj_id is not None:
            for obj in self.object_list:
                if obj.id == closest_obj_id:
                    obj.status = 'isgoal'
                    break

        res.obj_id = closest_obj_id
        res.obj_x = closest_obj_x
        res.obj_y = closest_obj_y
        res.obj_yaw = closest_obj_yaw
        
        return res
    
    def get_closest_box_callback(self, req, res):
        self.get_logger().info(f'get_closest_BOX_callback entered')
        closest_obj_id = None
        closest_obj_x = 0.0
        closest_obj_y = 0.0
        closest_obj_yaw = 0.0
        for obj in self.object_list:
            if obj.status == 'available' and obj.type == 'box':
                if closest_obj_id == None:
                    closest_obj_id = obj.id
                    closest_obj_x = obj.last_x
                    closest_obj_y = obj.last_y
                    closest_obj_yaw = obj.last_yaw
                    closest_distance = math.hypot(obj.last_x - req.robot_x, obj.last_y - req.robot_y)
                if math.hypot(obj.last_x - req.robot_x, obj.last_y - req.robot_y) < closest_distance:
                    closest_obj_id = obj.id
                    closest_obj_x = obj.last_x
                    closest_obj_y = obj.last_y
                    closest_obj_yaw = obj.last_yaw

        if closest_obj_id is not None:
            for obj in self.object_list:
                if obj.id == closest_obj_id:
                    obj.status = 'isgoal'
                    break

        res.obj_id = closest_obj_id
        res.obj_x = closest_obj_x
        res.obj_y = closest_obj_y
        res.obj_yaw = closest_obj_yaw
        
        return res
    
# -----------------------

    def set_status_callback(self, req, res):
        obj_id, status = req.obj_id, req.status

        for obj in self.object_list:
            if obj.id == obj_id:
                obj.status = status
        
        return res

# ---------------------

    def get_all_objects_callback(self, req, res):
        '''returns a list of all available objects (box and cubes)'''
        obj_pose_list = []
        for obj in self.object_list:
            if obj.status == 'available':
                obj_pose = ObjPose()
                obj_pose.obj_id = obj.id
                obj_pose.obj_type = obj.type
                obj_pose.obj_x = obj.last_x
                obj_pose.obj_y= obj.last_y
                obj_pose.obj_yaw = obj.last_yaw
                obj_pose_list.append(obj_pose)
        res.obj_poses = obj_pose_list
        return res        

# ------------------------

    def get_pos_of_obj_callback(self, req, res):
        
        for obj in self.object_list:
            if obj.id == req.obj_id:
                res.obj_x = obj.last_x
                res.obj_y = obj.last_y
                res.obj_yaw = obj.last_yaw
                return res

        self.get_logger().warning(f'Object with id {req.obj_id} not found in object_list during service call get_pos_of_obj')


# ------------------------

def main():
    rclpy.init()
    node = ObjectManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()
