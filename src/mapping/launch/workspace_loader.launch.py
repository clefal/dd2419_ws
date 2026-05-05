from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def _prefer_src_config(filename: str, pkg_share: str) -> str:
    # During development, prefer files in the workspace source tree.
    src_path = os.path.join(os.getcwd(), 'src', 'mapping', 'config', filename)
    if os.path.exists(src_path):
        return src_path
    return os.path.join(pkg_share, 'config', filename)


def generate_launch_description():

    pkg_share = get_package_share_directory('mapping')

    workspace_file = _prefer_src_config('final_workspace3.csv', pkg_share)
    map_file = _prefer_src_config('final_map3.csv', pkg_share)

    return LaunchDescription([
        DeclareLaunchArgument(
            'workspace_csv',
            default_value=workspace_file,
            description='Path to workspace polygon CSV',
        ),
        DeclareLaunchArgument(
            'map_csv',
            default_value=map_file,
            description='Path to map objects CSV',
        ),
        DeclareLaunchArgument(
            'input_units',
            default_value='cm',
            description='Units used in CSV files: m, cm, or mm',
        ),
        DeclareLaunchArgument(
            'publish_odom_frames',
            default_value='true',
            description='Publish static map->odom and map->odom_temp start-pose frames',
        ),

        # --- Mapping node ---
        Node(
            package='mapping',
            executable='workspace_loader',
            parameters=[{
                'workspace_csv': LaunchConfiguration('workspace_csv'),
                'map_csv': LaunchConfiguration('map_csv'),
                'input_units': LaunchConfiguration('input_units'),
                'publish_odom_frames': ParameterValue(
                    LaunchConfiguration('publish_odom_frames'),
                    value_type=bool,
                ),
            }]
        ),

        # --- Static TF: base_link -> lidar_link ---
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_lidar',
            arguments=[
                '--x', '0.0',
                '--y', '0.01',
                '--z', '0.08',
                '--frame-id', 'base_link',
                '--child-frame-id', 'lidar_link'
            ]
        ),


        # --- Static TF: base_link -> camera ---
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_camera',
            arguments=[
                '--x', '0.08987',
                '--y', '0.0175',
                '--z', '0.10456',
                '--qx', '0',
                '--qy', '0',
                '--qz', '0',
                '--qw', '1',
                '--frame-id', 'base_link',
                '--child-frame-id', 'realsense_camera_link'
            ]
        )
        
        # ,       
        #         # Static TF: base_link -> realsense_camera_depth_optical_frame
        # # this is only used for testing the detection_manager as it will not work if this transform is not put in place manually.
        # # When running things on the robot this transform is not needed as a transform from realsense_camera_link -> realsense...depth_optical_frame is put in place when launching the camera
        # Node(
        #     package='tf2_ros',
        #     executable='static_transform_publisher',
        #     name='base_to_camera',
        #     arguments=[
        #         '--x', '0.08987',
        #         '--y', '0.0175',
        #         '--z', '0.10456',
        #         '--qx', '0.5',
        #         '--qy', '-0.5',
        #         '--qz', '0.5',
        #         '--qw', '-0.5',
        #         '--frame-id', 'base_link',
        #         '--child-frame-id', 'realsense_camera_depth_optical_frame'
        #     ]
        # )
    ])
