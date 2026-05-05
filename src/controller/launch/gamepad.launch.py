from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        output='screen',
        parameters=[
            {
                'deadzone': 0.08,
                'autorepeat_rate': 20.0,
            }
        ],
    )

    gamepad_teleop = Node(
        package='controller',
        executable='gamepad_teleop',
        name='gamepad_teleop',
        output='screen',
        parameters=[
            {
                'forward_axis': 1,
                'turn_axis': 0,
                'deadman_button': 5,
                'stop_button': 1,
                'max_duty_cycle': 0.20,
                'boost_multiplier': 2.0,
                'boost_axis': 4,
                'boost_threshold': -0.5,
                'turn_scale': 0.75,
                'axis_deadzone': 0.08,
                'joy_timeout': 0.35,
                'publish_rate_hz': 20.0,
            }
        ],
    )

    return LaunchDescription([
        joy_node,
        gamepad_teleop,
    ])
