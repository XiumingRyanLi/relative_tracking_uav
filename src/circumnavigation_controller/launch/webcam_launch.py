from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='circumnavigation_controller',
            executable='bearing_measurement_generation',
            name='yolo_image_node',
            output='screen',
            parameters=[
                {'image_source': 'webcam', 'webcam_index': 0},
                {'show_debug_window': False},
                {'enable_debug_publish': False}
            ]
        ),
        Node(
            package='circumnavigation_controller',
            executable='controller',
            name='controller_node',
            output='screen'
        )
    ])
