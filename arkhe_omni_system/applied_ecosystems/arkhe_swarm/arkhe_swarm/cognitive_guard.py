# arkhe_omni_system/applied_ecosystems/arkhe_swarm/arkhe_swarm/cognitive_guard.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

class CognitiveGuard(Node):
    """
    Article 1/5: Cognitive Load Limit.
    Monitors handover frequency and reduces rate if ISC > 0.7.
    """
    def __init__(self):
        super().__init__('cognitive_guard')
        self.declare_parameter('max_load', 0.7)
        self.max_load = self.get_parameter('max_load').get_parameter_value().double_value

        self.current_load = {}

        # Subscribe to load topics for each drone
        for i in range(17):
            self.create_subscription(
                Float32,
                f'/drone{i}/load',
                lambda msg, idx=i: self.load_callback(msg, idx),
                10
            )

        # Reduction command publishers
        self.rate_pubs = []
        for i in range(17):
            pub = self.create_publisher(
                Float32,
                f'/drone{i}/target_rate',
                10
            )
            self.rate_pubs.append(pub)

        self.timer = self.create_timer(0.5, self.check_loads)

    def load_callback(self, msg, idx):
        self.current_load[idx] = msg.data

    def check_loads(self):
        for i in range(17):
            if i in self.current_load and self.current_load[i] > self.max_load:
                # Reduce handover rate by 50%
                reduction = Float32()
                reduction.data = 0.5  # Multiplicative factor
                self.rate_pubs[i].publish(reduction)
                self.get_logger().warn(f'Article 1 activated: drone{i} load={self.current_load[i]:.2f} exceeds threshold {self.max_load}')

def main(args=None):
    rclpy.init(args=args)
    node = CognitiveGuard()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
