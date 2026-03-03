# arkhe_omni_system/applied_ecosystems/arkhe_swarm/arkhe_swarm/arkhe_core.py
import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Float32, String
from geometry_msgs.msg import PoseStamped
import json

class ArkheCore(Node):
    """
    Central node for the Arkhe Drone Swarm.
    Coordinates global state, emergence monitoring, and constitutional integrity.
    """
    def __init__(self):
        super().__init__('arkhe_core')
        self.declare_parameter('n_drones', 17)
        self.n_drones = self.get_parameter('n_drones').get_parameter_value().integer_value

        self.drone_states = {}
        self.global_coherence = 0.0
        self.is_emergent = False

        # Subscriptions
        for i in range(self.n_drones):
            self.create_subscription(
                PoseStamped,
                f'/drone{i}/pose',
                lambda msg, idx=i: self.update_drone_pose(msg, idx),
                10
            )

        self.create_subscription(
            String,
            '/arkhe/global_coherence',
            self.update_coherence,
            10
        )

        self.create_subscription(
            String,
            '/arkhe/constitutional_violations',
            self.handle_violation,
            10
        )

        # Swarm Status Publisher
        self.status_pub = self.create_publisher(String, '/arkhe/swarm_status', 10)

        # Main loop timer
        self.timer = self.create_timer(2.0, self.publish_status)

        self.get_logger().info(f'Arkhe Core initialized for {self.n_drones} drones.')

    def update_drone_pose(self, msg, idx):
        self.drone_states[idx] = {
            'x': msg.pose.position.x,
            'y': msg.pose.position.y,
            'z': msg.pose.position.z
        }

    def update_coherence(self, msg):
        try:
            # Expecting JSON with C_global, C_local, emergence
            data = json.loads(msg.data)
            self.global_coherence = data.get('C_global', 0.0)
            self.is_emergent = bool(data.get('emergence', False))
        except Exception as e:
            self.get_logger().error(f"Failed to update coherence: {str(e)}")

    def handle_violation(self, msg):
        self.get_logger().error(f"CORE ALERT: {msg.data}")
        # Could trigger global safety protocols here

    def publish_status(self):
        status = {
            'n_drones_online': len(self.drone_states),
            'global_coherence': self.global_coherence,
            'emergence': self.is_emergent,
            'regime': self.calculate_regime()
        }
        msg = String()
        msg.data = json.dumps(status)
        self.status_pub.publish(msg)

    def calculate_regime(self):
        # Based on Article 2: Golden Ratio Threshold
        phi = 0.618033988749895
        z = self.global_coherence

        if z < phi * 0.7:
            return "DETERMINISTIC"
        elif phi * 0.7 <= z <= phi * 1.3:
            return "CRITICAL"
        else:
            return "STOCHASTIC"

def main(args=None):
    rclpy.init(args=args)
    node = ArkheCore()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
