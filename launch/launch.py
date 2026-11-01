import os
from launch import LaunchDescription
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.actions import ExecuteProcess, TimerAction
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    # CLOUD FILTER
    cloud_filter_node = ComposableNode(
        package='prcp_cloud_filter',
        plugin='cloud_filter_tmp', 
        name='prcp_cloud_filter',
        extra_arguments=[{'use_intra_process_comms': True}],
    )

    # KONTÉNER
    os_container = ComposableNodeContainer(
        name='os_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            cloud_filter_node,
        ],
        output='log'
    )

    # GROUND SEGMENTATION NODE
    ground_segmentation_node = Node(
        package='ground_segmentation_ros2',
        executable='ground_segmentation_ros2_node', 
        name='ground_segmentation',
        output='screen',
        parameters=[{
            'robot_frame': 'os_lidar',
            'maxX': 100.0,
            'minX': -100.0,
            'maxY': 100.0,
            'minY': -100.0,
            'maxZ': 100.0,
            'minZ': -100.0,
            'downsample': False,
            'downsample_resolution': 0.03,
            'lidar_to_ground': -1.129,
            'transform_tolerance': 0.1,
            'cellSizeX': 0.25,
            'cellSizeY': 0.25,
            'cellSizeZ': 10.0,
            'cellSizeZPhase2': 1.0,
            'slopeThresholdDegrees': 25.0,
            'groundInlierThreshold': 0.05,
            'centroidSearchRadius': 5.0,
            'use_imu_orientation': False,
            'maxGroundHeightDeviation': 0.1,
            'show_benchmark': True,
            'use_sim_time': True
        }]
    )

    # KAMERA GRID (Area)
    grid_cam_area_node = Node(
        package='prcp_grid_cam',
        executable='prcp_grid_cam_node',
        name='prcp_gridarea_cam',
        output='screen',
        parameters=[{
            'camera_info_topic': '/zed/zed_node/rgb/color/rect/image/camera_info',
            'camera_stream_topic': '/yolopx/mask',
            'target_color': 'red',
            'publish_topic_name': '/cam/grid/drivable_area',
            'target_frame_id': 'zed_camera_front',
            'grid_fill_value': 0,         
            'grid_background_value': -1,  
            'position_x': 0.0, 
            'position_y': 0.0,
            'length_x': 100.0, 
            'length_y': 100.0,
            'cell_size': 0.15,
            'camera_horizon_offset': 0.0,
            'max_distance_x': 40.0,
            'max_distance_y': 20.0,
            'pitch_compensation_deg': 0.0,
            
            'use_sim_time': True
        }]
    )

    # KAMERA GRID (Lanes)
    grid_cam_lanes_node = Node(
        package='prcp_grid_cam',
        executable='prcp_grid_cam_node',
        name='prcp_gridlanes_cam',
        output='screen',
        parameters=[{
            'camera_info_topic': '/zed/zed_node/rgb/color/rect/image/camera_info',
            'camera_stream_topic': '/yolopx/mask',
            'target_color': 'green',
            'publish_topic_name': '/cam/grid/bad_lines',
            'target_frame_id': 'zed_camera_front',
            'grid_fill_value': 100,
            'grid_background_value': -1,
            'position_x': 0.0, 
            'position_y': 0.0,
            'length_x': 100.0, 
            'length_y': 100.0,
            'cell_size': 0.15,
            'camera_horizon_offset': 0.0,
            'max_distance_x': 40.0,
            'max_distance_y': 20.0,
            'pitch_compensation_deg': 0.0,

            'use_sim_time': True
        }]
    )

    # GRID FÚZIÓ
    grid_fusion_node = Node(
        package='prcp_grid_fusion',
        executable='prcp_grid_fusion_node',
        name='prcp_grid_fusion',
        output='screen',
        parameters=[{
            'cell_size': 0.15, 
            'length_x': 50.0,
            'length_y': 70.0,
            'distance_covered_confidence': 2.0,

            'exclusion_box_x_min': 0.0,
            'exclusion_box_x_max': 0.4,
            'exclusion_box_y_min': -0.3,
            'exclusion_box_y_max': 0.3,
            'freespace_box_x_min': 0.0,
            'freespace_box_x_max': 2.0,
            'freespace_box_y_min': -1.45,
            'freespace_box_y_max': 1.45,

            'drivable_closing_kernel_size': 100,
            'obstacle_points_height_min': 0.17,
            'obstacle_points_height_max': 1.5,
            'position_x': 25.0,
            'position_y': 0.0,
            'force_closing_no_temporal': True,
            'enable_road_blur': False,
            'enable_obstacle_blur': True,
            'enable_lanes_blur': True,
            'drivable_mask_threshold': 85,
            'use_sim_time': True,

            # Phase 1
            'logs_odds_increase_camera': 2.0, 
            'obstacle_fill_weight': 4.0,
            'lanes_fill_weight': 2.3,
            'blur_kernel_x': 1.0,
            'blur_kernel_y': 1.0, 
            'obstacle_blur_kernel_x': 1.5,
            'obstacle_blur_kernel_y': 1.5,
            'lanes_blur_kernel_x': 2.0,
            'lanes_blur_kernel_y': 2.0,
            'ror_radius': 0.5,
            'ror_min_neighbors': 1,

            # Távolság
            'distance_threshold': 1.0,
            
            # Phase 2
            'logs_odds_increase_camera_phase2': 2.0, 
            'obstacle_fill_weight_phase2': 80.0,
            'lanes_fill_weight_phase2': 3.3,
            'blur_kernel_x_phase2': 0.5,
            'blur_kernel_y_phase2': 0.5,
            'obstacle_blur_kernel_x_phase2': 18.0,
            'obstacle_blur_kernel_y_phase2': 18.0,
            'lanes_blur_kernel_x_phase2': 8.0,
            'lanes_blur_kernel_y_phase2': 8.0,
            'ror_radius_phase2': 0.5,
            'ror_min_neighbors_phase2': 1,
            'enable_road_blur_phase2': False,
            'enable_obstacle_blur_phase2': True,
            'enable_lanes_blur_phase2': True,
        }]
    )

    # dummy_pub_start = ExecuteProcess(
    #     cmd=[
    #         'ros2', 'topic', 'pub', '--once', 
    #         '/move_base_simple/goal', 'geometry_msgs/msg/PoseStamped', 
    #         "{header: {frame_id: 'base_link'}, pose: {position: {x: 0.0, y: 0.0, z: 0.0}}}"
    #     ],
    #     output='screen'
    # )

    # dummy_pub_end = TimerAction(
    #     period=1.0, 
    #     actions=[
    #         ExecuteProcess(
    #             cmd=[
    #                 'ros2', 'topic', 'pub', '--once', 
    #                 '/move_base_simple/goal', 'geometry_msgs/msg/PoseStamped', 
    #                 "{header: {frame_id: 'base_link'}, pose: {position: {x: 5.0, y: 0.0, z: 0.0}}}"
    #             ],
    #             output='screen'
    #         )
    #     ]
    # )

    # static_tf_base_to_sensor = Node(
    #     package='tf2_ros',
    #     executable='static_transform_publisher',
    #     name='static_tf_base_to_sensor',
    #     arguments=['--x', '0.0', '--y', '0.0', '--z', '1.129', '--yaw', '1.5708', '--pitch', '0.0', '--roll', '0.0', '--frame-id', 'base_link', '--child-frame-id', 'os_sensor']
    # )

    # static_tf_sensor_to_lidar = Node(
    #     package='tf2_ros',
    #     executable='static_transform_publisher',
    #     name='static_tf_sensor_to_lidar',
    #     arguments=['--x', '0.0', '--y', '0.0', '--z', '0.03618', '--yaw', '3.14159', '--pitch', '0.0', '--roll', '0.0', '--frame-id', 'os_sensor', '--child-frame-id', 'os_lidar']
    # )

    # static_tf_base_to_camera = Node(
    #     package='tf2_ros',
    #     executable='static_transform_publisher',
    #     name='static_tf_base_to_camera',
    #     arguments=['--x', '1.0', '--y', '0.0', '--z', '1.3', '--yaw', '0.0', '--pitch', '0.0', '--roll', '0.0', '--frame-id', 'base_link', '--child-frame-id', 'zed_camera_front']
    # )

    # static_tf_base_to_ground = Node(
    #     package='tf2_ros',
    #     executable='static_transform_publisher',
    #     name='static_tf_base_to_ground',
    #     arguments=['--x', '0.0', '--y', '0.0', '--z', '0.0', '--yaw', '0.0', '--pitch', '0.0', '--roll', '0.0', '--frame-id', 'base_link', '--child-frame-id', 'ground_link']
    # )

    return LaunchDescription([
        os_container,
        ground_segmentation_node,
        grid_cam_area_node,
        grid_cam_lanes_node,
        grid_fusion_node,
        # dummy_pub_start,
        # dummy_pub_end,
        # static_tf_base_to_sensor,
        # static_tf_sensor_to_lidar,
        # static_tf_base_to_camera,
        # static_tf_base_to_ground,
    ])
