
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    pkg_share = get_package_share_directory('mapping')

    workspace_file = os.path.join(pkg_share, 'config', 'workspace_1.csv')
    map_file = os.path.join(pkg_share, 'config', 'map_1_1.csv')

    return LaunchDescription([
        Node(
            package='mapping',
            executable='workspace_loader',
            parameters=[
                {
                    'workspace_csv': workspace_file,
                    'map_csv': map_file,
                    'input_units': 'cm'
                }
            ]
        )
    ])
