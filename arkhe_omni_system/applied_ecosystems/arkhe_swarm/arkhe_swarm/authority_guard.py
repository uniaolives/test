# arkhe_omni_system/applied_ecosystems/arkhe_swarm/arkhe_swarm/authority_guard.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import json

class AuthorityGuard(Node):
    """
    Article 3/7: Human Final Authority.
    Intercepts critical actions and requires human validation.
    """
    def __init__(self):
        super().__init__('authority_guard')
        self.declare_parameter('critical_actions', ["land", "fire", "change_mission"])
        self.critical_actions = self.get_parameter('critical_actions').get_parameter_value().string_array_value

        self.pending_approvals = {}

        # Subscribe to action requests
        self.create_subscription(
            String,
            '/arkhe/action_requests',
            self.request_callback,
            10
        )

        # Subscribe to human approvals
        self.create_subscription(
            String,
            '/arkhe/human_approvals',
            self.approval_callback,
            10
        )

        # Publisher for authorized actions
        self.authorized_pub = self.create_publisher(
            String,
            '/arkhe/authorized_actions',
            10
        )

        self.get_logger().info(f'AuthorityGuard (Article 3/7) monitoring critical actions: {self.critical_actions}')

    def request_callback(self, msg):
        try:
            request = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        action = request.get('action')
        request_id = request.get('id')

        if action in self.critical_actions:
            self.get_logger().warn(f"Critical action '{action}' (ID: {request_id}) requested. Awaiting HUMAN APPROVAL.")
            self.pending_approvals[request_id] = request
            # In a real system, this would trigger a notification to the human operator
        else:
            # Non-critical actions can be authorized immediately or pass through
            self.get_logger().info(f"Non-critical action '{action}' authorized automatically.")
            self.authorized_pub.publish(msg)

    def approval_callback(self, msg):
        try:
            approval = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        request_id = approval.get('id')
        approved = approval.get('approved', False)

        if request_id in self.pending_approvals:
            request = self.pending_approvals.pop(request_id)
            if approved:
                self.get_logger().info(f"Action '{request.get('action')}' (ID: {request_id}) APPROVED by human.")
                auth_msg = String()
                auth_msg.data = json.dumps(request)
                self.authorized_pub.publish(auth_msg)
            else:
                self.get_logger().error(f"Action '{request.get('action')}' (ID: {request_id}) REJECTED by human.")
        else:
            self.get_logger().error(f"Received approval for unknown request ID: {request_id}")

def main(args=None):
    rclpy.init(args=args)
    node = AuthorityGuard()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
