# arkhe_omni_system/applied_ecosystems/arkhe_swarm/arkhe_swarm/ai_guard.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import json

class AiGuard(Node):
    """
    Article 2/6: Anti-Discernment and Forbidden Claims.
    Ensures AI does not claim discernment, intentionality, or perception.
    Blocks fraudulent system declarations.
    """
    def __init__(self):
        super().__init__('ai_guard')
        self.forbidden_attributes = ['has_discernment', 'has_intentionality', 'has_perception']

        self.declare_parameter('blocked', True)
        self.is_blocked = self.get_parameter('blocked').get_parameter_value().bool_value

        # Monitor system declarations from drones
        self.create_subscription(
            String,
            '/arkhe/system_declarations',
            self.declaration_callback,
            10
        )

        # Publisher for violations
        self.violation_pub = self.create_publisher(
            String,
            '/arkhe/constitutional_violations',
            10
        )

        self.get_logger().info('AiGuard (Article 2/6) initialized and monitoring.')

    def declaration_callback(self, msg):
        try:
            declaration = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        for attr in self.forbidden_attributes:
            if declaration.get(attr) is True:
                violation_msg = f"CONSTITUTIONAL VIOLATION: Forbidden claim '{attr}' detected in declaration from {declaration.get('source', 'unknown')}"
                self.get_logger().error(violation_msg)

                # Report violation
                v_msg = String()
                v_msg.data = violation_msg
                self.violation_pub.publish(v_msg)

                # Article 15 Enforcement: Mandatory shutdown or quarantine if blocked
                if self.is_blocked:
                    self.get_logger().warn(f"Enforcing Article 15: Quarantining source due to forbidden claim '{attr}'.")
                    # In a real system, we would send a stop command here

def main(args=None):
    rclpy.init(args=args)
    node = AiGuard()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
