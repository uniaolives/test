# arkhe_omni_system/applied_ecosystems/arkhe_swarm/launch/arkhe_twin_cities.launch.py
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, LogInfo
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('arkhe_swarm')
    config_file = os.path.join(pkg_share, 'config', 'arkhe_twin_cities.yaml')
    world_file = os.path.join(pkg_share, 'worlds', 'twin_cities.sdf')

    # Micro XRCE-DDS Agent
    uxrce_agent = ExecuteProcess(
        cmd=['MicroXRCEAgent', 'udp4', '-p', '8888'],
        output='screen'
    )

    # Gazebo Sim
    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', '-r', world_file],
        output='screen'
    )

    # Arkhe Core
    arkhe_core = Node(
        package='arkhe_swarm',
        executable='arkhe_core',
        parameters=[config_file],
        output='screen'
    )

    # GHZ Consensus
    ghz_consensus = Node(
        package='arkhe_swarm',
        executable='ghz_consensus',
        parameters=[config_file],
        output='screen'
    )

    # Simulation Data Provider
    drone_sim = Node(
        package='arkhe_swarm',
        executable='drone_state_sim',
        parameters=[config_file],
        output='screen'
    )

    # Pleroma Kernel
    pleroma_kernel = Node(
        package='arkhe_swarm',
        executable='pleroma_kernel',
        parameters=[config_file],
        output='screen'
    )

    # Constitutional Guards
    guards = [
        Node(package='arkhe_swarm', executable='cognitive_guard', parameters=[config_file]),
        Node(package='arkhe_swarm', executable='ai_guard', parameters=[config_file]),
        Node(package='arkhe_swarm', executable='authority_guard', parameters=[config_file]),
        Node(package='arkhe_swarm', executable='transparency_guard', parameters=[config_file]),
    ]

    # Drone Spawning (17 drones: 8 Rio, 8 SP, 1 Bridge)
    spawn_nodes = []
    for i in range(17):
        # Hyperbolic positions for Twin Cities configuration (Option B: 3D)
        if i < 8: # Rio
            x = -2.0 + (i % 3) * 0.5
            y = 0.2 + (i // 3) * 0.2
            z = 0.1 + (i % 2) * 500.0 # Vary altitude to simulate atmospheric layers
        elif i < 16: # SP
            x = 2.0 + ((i-8) % 3) * 0.5
            y = 0.2 + ((i-8) // 3) * 0.2
            z = 0.1 + ((i-8) % 2) * 1000.0
        else: # Bridge
            x = 0.0
            y = 0.5
            z = 2000.0 # Bridge at higher altitude
    # Drone Spawning (17 drones)
    spawn_nodes = []
    for i in range(17):
        # Calculate hyperbolic position (mock)
        x = float(i % 4) * 2.0
        y = float(i // 4) * 2.0 + 1.0 # Ensure y > 0

        spawn_nodes.append(
            Node(
                package='gazebo_ros',
                executable='spawn_entity.py',
                arguments=[
                    '-entity', f'iris_{i}',
                    '-x', str(x),
                    '-y', str(y),
                    '-z', str(z),
                    '-z', '0.1',
                    '-robot_namespace', f'drone{i}'
                ],
                output='screen'
            )
        )

    return LaunchDescription([
        uxrce_agent,
        gazebo,
        arkhe_core,
        ghz_consensus,
        drone_sim,
        pleroma_kernel,
        *guards,
        *spawn_nodes
    ])
