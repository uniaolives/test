# arkhe_omni_system/applied_ecosystems/arkhe_swarm/arkhe_swarm/transparency_guard.py
import rclpy
from rclpy.node import Node
import json
import hashlib
import time
import os
from std_msgs.msg import String

class TransparencyGuard(Node):
    """
    Article 4/9: Transparency and Auditability.
    Registers all handovers in an immutable local ledger.
    """
    def __init__(self):
        super().__init__('transparency_guard')
        self.declare_parameter('ledger_path', '/tmp/arkhe/drone_ledger.json')
        self.ledger_file = self.get_parameter('ledger_path').get_parameter_value().string_value

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.ledger_file), exist_ok=True)

        self.ledger = []
        if os.path.exists(self.ledger_file):
            try:
                with open(self.ledger_file, 'r') as f:
                    self.ledger = json.load(f)
            except Exception:
                self.ledger = []

        self.subscription = self.create_subscription(
            String,
            '/arkhe/handover_log',
            self.log_callback,
            10
        )

        self.get_logger().info(f'TransparencyGuard initialized. Ledger at {self.ledger_file}')

    def log_callback(self, msg):
        try:
            entry = json.loads(msg.data)
        except json.JSONDecodeError:
            entry = {"raw_data": msg.data}

        entry['timestamp'] = time.time()

        # Chaining hash (Immutable pattern)
        previous_hash = self.ledger[-1]['hash'] if self.ledger else "GENESIS"
        entry['previous_hash'] = previous_hash

        entry_content = json.dumps(entry, sort_keys=True).encode()
        entry['hash'] = hashlib.sha256(entry_content).hexdigest()

        self.ledger.append(entry)

        # Periodic save (every 10 entries)
        if len(self.ledger) % 10 == 0:
            self.save_ledger()
            self.get_logger().info(f'Ledger updated: {len(self.ledger)} entries')

    def save_ledger(self):
        try:
            with open(self.ledger_file, 'w') as f:
                json.dump(self.ledger, f, indent=2)
        except Exception as e:
            self.get_logger().error(f'Failed to save ledger: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = TransparencyGuard()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
