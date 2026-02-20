from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    pkg_share = get_package_share_directory('mapping')

    workspace_file = os.path.join(pkg_share, 'config', 'workspace_1.csv')
    map_file = os.path.join(pkg_share, 'config', 'map_1_1.csv')

    return LaunchDescription([

        # --- Mapping node ---
        Node(
            package='mapping',
            executable='workspace_loader',
            parameters=[{
                'workspace_csv': workspace_file,
                'map_csv': map_file,
                'input_units': 'cm'
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
                '--qx', '0.5',
                '--qy', '-0.5',
                '--qz', '0.5',
                '--qw', '-0.5',
                '--frame-id', 'base_link',
                '--child-frame-id', 'realsense_camera_depth_optical_frame'
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
                '--qx', '0.5',
                '--qy', '-0.5',
                '--qz', '0.5',
                '--qw', '-0.5',
                '--frame-id', 'base_link',
                '--child-frame-id', 'realsense_camera_link'
            ]
        )
    ])
