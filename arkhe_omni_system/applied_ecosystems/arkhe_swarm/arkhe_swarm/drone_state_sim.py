# arkhe_omni_system/applied_ecosystems/arkhe_swarm/arkhe_swarm/drone_state_sim.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray, String
import json
import random
import numpy as np

class DroneStateSim(Node):
    """
    Simulates internal state, load, and sensor data (including THz readings) for the swarm.
    """
    def __init__(self):
        super().__init__('drone_state_sim')
        self.declare_parameter('n_drones', 17)
        self.n_drones = self.get_parameter('n_drones').get_parameter_value().integer_value

        self.state_pubs = []
        self.load_pubs = []
        self.thz_pubs = [] # Simulated THz sensor readings (Task 1.3)

        for i in range(self.n_drones):
            self.state_pubs.append(self.create_publisher(Float32MultiArray, f'/drone{i}/state', 10))
            self.load_pubs.append(self.create_publisher(Float32, f'/drone{i}/load', 10))
            self.thz_pubs.append(self.create_publisher(Float32, f'/drone{i}/thz_reading', 10))

        self.timer = self.create_timer(1.0, self.publish_data)
        self.get_logger().info(f'Drone State Simulator started for {self.n_drones} drones.')

    def publish_data(self):
        for i in range(self.n_drones):
            # 1. State: simulating 0 or 1 for GHZ consensus
            # Most drones should align to demonstrate coherence
            base_state = 1 if random.random() < 0.8 else 0
            state_msg = Float32MultiArray()
            state_msg.data = [float(base_state)]
            self.state_pubs[i].publish(state_msg)

            # 2. Load: simulating cognitive load (Article 1/5)
            # Occasional spikes to test CognitiveGuard
            load = 0.3 + (0.5 * random.random() if random.random() > 0.9 else 0.1)
            load_msg = Float32()
            load_msg.data = float(load)
            self.load_pubs[i].publish(load_msg)

            # 3. THz Sensor Simulation (Task 1.3)
            # Simulated analyte detection (e.g., peak at 1.5 THz)
            # Based on Fu et al. - simplified as a noisy baseline with occasional "hits"
            hit = 10.0 if random.random() > 0.95 else 1.0
            thz_val = hit + random.gauss(0, 0.1)
            thz_msg = Float32()
            thz_msg.data = float(thz_val)
            self.thz_pubs[i].publish(thz_msg)

def main(args=None):
    rclpy.init(args=args)
    node = DroneStateSim()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
