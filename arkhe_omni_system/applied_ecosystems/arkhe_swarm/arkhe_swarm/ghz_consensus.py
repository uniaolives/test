# arkhe_omni_system/applied_ecosystems/arkhe_swarm/arkhe_swarm/ghz_consensus.py
import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Float32MultiArray, String
from geometry_msgs.msg import PoseStamped
import json
from .ppp_utils import hyperbolic_distance_uhp, check_q_process_condition

class GHZConsensus(Node):
    """
    Implements multi-party GHZ entanglement for the swarm.
    Publishes C_global and detects emergence.

    Based on Graph Diffusion Network for MARL.
    """

    def __init__(self):
        super().__init__('ghz_consensus')
        self.declare_parameter('n_drones', 17)
        self.declare_parameter('tau', 2.0)
        self.declare_parameter('v_max', 0.005)

        self.n_drones = self.get_parameter('n_drones').get_parameter_value().integer_value
        self.tau = self.get_parameter('tau').get_parameter_value().double_value
        self.v_max = self.get_parameter('v_max').get_parameter_value().double_value

        self.n_drones = self.get_parameter('n_drones').get_parameter_value().integer_value
        self.tau = self.get_parameter('tau').get_parameter_value().double_value

        self.positions = {}
        self.states = {}  # simulated quantum states (0 or 1)

        # Subscriptions for each drone
        for i in range(self.n_drones):
            self.create_subscription(
                PoseStamped,
                f'/drone{i}/pose',
                lambda msg, idx=i: self.pose_callback(msg, idx),
                10
            )
            self.create_subscription(
                Float32MultiArray,
                f'/drone{i}/state',
                lambda msg, idx=i: self.state_callback(msg, idx),
                10
            )

        # Global coherence publisher
        self.coherence_pub = self.create_publisher(
            String,
            '/arkhe/global_coherence',
            10
        )

        # Periodic computation timer
        self.timer = self.create_timer(1.0, self.compute_coherence)

    def pose_callback(self, msg, idx):
        # Support 2D and 3D. Vertical coordinate is always the last one.
        self.positions[idx] = np.array([msg.pose.position.x,
                                        msg.pose.position.y,
                                        msg.pose.position.z])
        self.positions[idx] = np.array([msg.pose.position.x,
                                        msg.pose.position.y])

    def state_callback(self, msg, idx):
        self.states[idx] = msg.data[0]  # 0 or 1

    def compute_coherence(self):
        if len(self.positions) < self.n_drones:
            return

        # Hyperbolic distance matrix (Upper Half-Plane Model)
        dist_matrix = np.zeros((self.n_drones, self.n_drones))
        for i in range(self.n_drones):
            for j in range(self.n_drones):
                if i != j:
                    dist_matrix[i,j] = hyperbolic_distance_uhp(self.positions[i], self.positions[j])
                    # Half-plane coordinates (ensure y > 0)
                    y_i = max(0.1, self.positions[i][1])
                    y_j = max(0.1, self.positions[j][1])

                    # Hyperbolic distance: dH = arcosh(1 + ((x_i-x_j)² + (y_i-y_j)²)/(2*y_i*y_j))
                    val = 1 + ((self.positions[i][0] - self.positions[j][0])**2 +
                                (y_i - y_j)**2) / (2 * y_i * y_j)
                    dist_matrix[i,j] = np.arccosh(max(1.0, val))

        # Probabilistic Handover
        handover_prob = np.exp(-dist_matrix / self.tau)
        # Simplified: check if states align according to GHZ fidelity

        if len(self.states) == self.n_drones:
            states_array = np.array([self.states[i] for i in range(self.n_drones)])
            # Ideal GHZ: all 0 or all 1
            prob_all_zero = np.mean(states_array == 0)
            prob_all_one = np.mean(states_array == 1)
            ghz_fidelity = prob_all_zero + prob_all_one - 1.0  # normalized [0,1]

            # Average local coherence
            C_local = np.mean([s for s in self.states.values()])

            # Emergence: C_global > max(C_local)
            C_global = max(0.0, ghz_fidelity)
            emergence = C_global > C_local

            # Stability Check (Q-process condition)
            # Simplified neighbor count: all drones (global check)
            n_neighbors = self.n_drones
            stable = check_q_process_condition(self.v_max, n_neighbors, d=3)

            # Publish metrics
            msg = String()
            msg.data = json.dumps({
                'C_global': float(C_global),
                'C_local': float(C_local),
                'emergence': bool(emergence),
                'stable': bool(stable)
                'emergence': bool(emergence)
            })
            self.coherence_pub.publish(msg)

            self.get_logger().info(f'C_global: {C_global:.3f}, C_local: {C_local:.3f}, Emergence: {emergence}')

def main(args=None):
    rclpy.init(args=args)
    node = GHZConsensus()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
