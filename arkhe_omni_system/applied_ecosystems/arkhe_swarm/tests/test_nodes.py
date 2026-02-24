# arkhe_omni_system/applied_ecosystems/arkhe_swarm/tests/test_nodes.py
import sys
import unittest
from unittest.mock import MagicMock, patch
import json
import numpy as np

# Mock rclpy before importing nodes
sys.modules['rclpy'] = MagicMock()
sys.modules['rclpy.node'] = MagicMock()
sys.modules['std_msgs.msg'] = MagicMock()
sys.modules['geometry_msgs.msg'] = MagicMock()

# Now we can import the classes, but we need to handle the inheritance
import rclpy
from rclpy.node import Node

# Mocking Node for testing
class MockNode:
    def __init__(self, name):
        self.name = name
        self.subscriptions = []
        self.publishers = {}
        self.parameters = {}

    def create_subscription(self, msg_type, topic, callback, qos):
        self.subscriptions.append({'topic': topic, 'callback': callback})

    def create_publisher(self, msg_type, topic, qos):
        pub = MagicMock()
        self.publishers[topic] = pub
        return pub

    def create_timer(self, period, callback):
        return MagicMock()

    def declare_parameter(self, name, value):
        self.parameters[name] = MagicMock(get_parameter_value=lambda: MagicMock(integer_value=value, double_value=value, bool_value=value, string_value=value, string_array_value=value))

    def get_parameter(self, name):
        return self.parameters[name]

    def get_logger(self):
        return MagicMock()

# Patch the Node class in the modules
with patch('rclpy.node.Node', MockNode):
    from arkhe_swarm.ai_guard import AiGuard
    from arkhe_swarm.authority_guard import AuthorityGuard
    from arkhe_swarm.cognitive_guard import CognitiveGuard
    from arkhe_swarm.arkhe_core import ArkheCore
    from arkhe_swarm.ghz_consensus import GHZConsensus
    from arkhe_swarm.drone_state_sim import DroneStateSim
    from arkhe_swarm.pleroma_kernel import PleromaKernel, QuantumState
    from arkhe_swarm.ppp_utils import hyperbolic_distance_uhp, sample_ppp_hyperbolic, atmospheric_density, stability_threshold_q_process

class TestArkheGuards(unittest.TestCase):
    def test_ai_guard_violation(self):
        with patch('rclpy.node.Node', MockNode):
            guard = AiGuard()

            # Simulate a message with forbidden claim
            msg = MagicMock()
            msg.data = json.dumps({
                'source': 'drone0',
                'has_discernment': True
            })

            # Call callback
            guard.declaration_callback(msg)

            # Check if violation was published
            self.assertTrue(guard.publishers['/arkhe/constitutional_violations'].publish.called)
            call_args = guard.publishers['/arkhe/constitutional_violations'].publish.call_args[0][0]
            self.assertIn("Forbidden claim 'has_discernment' detected", call_args.data)

    def test_authority_guard_intercept(self):
        with patch('rclpy.node.Node', MockNode):
            guard = AuthorityGuard()

            # Simulate a critical action request
            msg = MagicMock()
            msg.data = json.dumps({
                'id': 'req123',
                'action': 'fire'
            })

            guard.request_callback(msg)

            # Should NOT be published immediately
            self.assertFalse(guard.publishers['/arkhe/authorized_actions'].publish.called)
            self.assertIn('req123', guard.pending_approvals)

            # Simulate human approval
            app_msg = MagicMock()
            app_msg.data = json.dumps({
                'id': 'req123',
                'approved': True
            })

            guard.approval_callback(app_msg)

            # Now it should be published
            self.assertTrue(guard.publishers['/arkhe/authorized_actions'].publish.called)
            auth_call = guard.publishers['/arkhe/authorized_actions'].publish.call_args[0][0]
            self.assertEqual(json.loads(auth_call.data)['id'], 'req123')

    def test_cognitive_guard_threshold(self):
        with patch('rclpy.node.Node', MockNode):
            guard = CognitiveGuard()

            # Simulate high load
            msg = MagicMock()
            msg.data = 0.8  # > 0.7

            guard.load_callback(msg, 0)
            guard.check_loads()

            # Check if reduction was published to drone0/target_rate
            # Since we created 17 publishers, index 0 is at publishers['/drone0/target_rate']
            # Wait, how did I store publishers in MockNode?
            # In cognitive_guard.py:
            # self.rate_pubs.append(pub)
            # So I should check guard.rate_pubs[0].publish.called

            self.assertTrue(guard.rate_pubs[0].publish.called)
            self.assertEqual(guard.rate_pubs[0].publish.call_args[0][0].data, 0.5)

    def test_arkhe_core_json_parsing(self):
        with patch('rclpy.node.Node', MockNode):
            core = ArkheCore()

            # Simulate JSON coherence message
            msg = MagicMock()
            msg.data = json.dumps({
                'C_global': 0.618,
                'C_local': 0.4,
                'emergence': True,
                'stable': True
                'emergence': True
            })

            core.update_coherence(msg)

            self.assertEqual(core.global_coherence, 0.618)
            self.assertTrue(core.is_emergent)

    def test_ghz_consensus_3d_stability(self):
        with patch('rclpy.node.Node', MockNode):
            node = GHZConsensus()
            node.n_drones = 2

            # Mock 3D positions
            msg1 = MagicMock()
            msg1.pose.position.x = 0.0
            msg1.pose.position.y = 0.0
            msg1.pose.position.z = 1.0
            node.pose_callback(msg1, 0)

            msg2 = MagicMock()
            msg2.pose.position.x = 0.0
            msg2.pose.position.y = 0.0
            msg2.pose.position.z = 2.0
            node.pose_callback(msg2, 1)

            # Mock states
            s_msg = MagicMock()
            s_msg.data = [1.0]
            node.state_callback(s_msg, 0)
            node.state_callback(s_msg, 1)

            node.compute_coherence()

            # Check published metrics
            self.assertTrue(node.coherence_pub.publish.called)
            data = json.loads(node.coherence_pub.publish.call_args[0][0].data)
            self.assertIn('stable', data)
            # Threshold for d=3 is 0.5. v_max * n_drones = 0.005 * 2 = 0.01 < 0.5. Should be stable.
            self.assertTrue(data['stable'])

    def test_ppp_utils_distance_2d(self):
        p1 = (0.0, 1.0)
        p2 = (0.0, 2.0)
        # dH = arcosh(1 + (0^2 + 1^2)/(2*1*2)) = arcosh(1 + 0.25) = arcosh(1.25)
        dist = hyperbolic_distance_uhp(p1, p2)
        self.assertAlmostEqual(dist, 0.693147, places=5) # ln(1.25 + sqrt(1.25^2 - 1)) approx 0.693

    def test_ppp_utils_distance_3d(self):
        p1 = (0.0, 0.0, 1.0)
        p2 = (0.0, 0.0, 2.0)
        dist = hyperbolic_distance_uhp(p1, p2)
        self.assertAlmostEqual(dist, 0.693147, places=5)

    def test_atmospheric_density(self):
        self.assertEqual(atmospheric_density(0), 1.225)
        self.assertLess(atmospheric_density(8500), 1.225 / 2.0) # exp(-1) approx 0.36

    def test_stability_thresholds(self):
        self.assertEqual(stability_threshold_q_process(2), 0.125)
        self.assertEqual(stability_threshold_q_process(3), 0.5)

    def test_drone_state_sim_publish(self):
        with patch('rclpy.node.Node', MockNode):
            sim = DroneStateSim()
            sim.publish_data()

            # Check if topics were published
            self.assertTrue(sim.state_pubs[0].publish.called)
            self.assertTrue(sim.load_pubs[0].publish.called)
            self.assertTrue(sim.thz_pubs[0].publish.called)

    def test_quantum_state_evolution(self):
        qs = QuantumState(max_n=2, max_m=2)
        initial_amp = qs.amplitudes[0, 0]
        self.assertEqual(initial_amp, 1.0)

        qs.evolve(dt=0.1, hbar=1.0)
        # Norm should be preserved
        norm = np.sum(np.abs(qs.amplitudes)**2)
        self.assertAlmostEqual(norm, 1.0)

    def test_pleroma_kernel_loop(self):
        with patch('rclpy.node.Node', MockNode):
            kernel = PleromaKernel()
            kernel.neighbor_states = {
                0: {'pose': [0, 0, 1]},
                1: {'pose': [0, 0, 2]}
            }
            kernel.main_loop()

            # Check if coherence was published
            self.assertTrue(kernel.coherence_pub.publish.called)
            self.assertTrue(kernel.state_pub.publish.called)

if __name__ == '__main__':
    unittest.main()
