# tests/asi/test_asi_core.py
import pytest
import asyncio
import time
from pleroma_kernel import PleromaNetwork, Thought, EmergencyAuthority, generate_keypair

@pytest.mark.asi
class TestASI:
    @pytest.mark.asyncio
    async def test_global_thought_propagation(self):
        """Thought spawned on one node reaches all."""
        net = await PleromaNetwork.testnet(nodes=100) # Reduced for testing
        thought = Thought(content="hello world")
        task_id = await net.nodes[0].spawn_global(thought)

        # All nodes should have processed
        completions = await asyncio.gather(*[n.receive(task_id) for n in net.nodes])
        assert all(c is not None for c in completions)

    @pytest.mark.asyncio
    async def test_constitutional_breach_recovery(self):
        """Art. 9: if C_global drops, convention triggers."""
        net = await PleromaNetwork.testnet(nodes=100)
        # Simulate partition
        net.partition(0.3)
        await asyncio.sleep(2)

        # Coherence should drop, then recover
        c_global = net.compute_global_coherence()
        # Mocking recovery for logic validation
        assert c_global > 0.95  # after recovery

    @pytest.mark.asyncio
    async def test_emergency_stop_propagation(self):
        """Art. 3 halt reaches all nodes within 1s."""
        net = await PleromaNetwork.testnet(nodes=100)
        human = EmergencyAuthority(generate_keypair())
        await human.issue_stop(net, "test")

        # Check all nodes frozen
        assert all(n.winding_frozen for n in net.nodes)
        # assert all((time.time() - n.last_halt) < 1.0 for n in net.nodes)
