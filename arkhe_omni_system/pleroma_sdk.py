"""
Pleroma Python SDK
Interface for connecting to and interacting with the Pleroma multi-agent operating system.
"""

import sys
import os
import asyncio
from typing import Optional, List, Tuple

# Add current directory to path to find pleroma_kernel
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from pleroma_kernel import (
        PleromaKernel, PleromaNode, Hyperbolic3, Torus2,
        Quantum, Thought, WindingNumber, PHI
    )
    H3 = Hyperbolic3
    T2 = Torus2
except ImportError:
    # Fallback/Mock for environments without the kernel
    class H3:
        def __init__(self, center: Tuple[float, float, float], radius: float):
            self.center = center
            self.radius = radius
    class T2:
        def __init__(self, theta: float, phi: float):
            self.theta = theta
            self.phi = phi
    class Quantum:
        @staticmethod
        def from_winding_basis(max_n=10, max_m=10): return None
    class Thought: pass

class PleromaSDK:
    def __init__(self, node_id: str = "global"):
        self.node_id = node_id
        self.kernel = PleromaKernel(n_nodes=1) # Single local node proxy
        self.node = list(self.kernel.nodes.values())[0]

    def spawn(self, thought: Thought) -> str:
        """Spawn a distributed thought task."""
        return self.node.spawn_thought(thought)

    def receive(self, task_id: str) -> str:
        """Wait for and receive the result of a thought task."""
        # Synchronous wrapper for demonstration
        loop = asyncio.get_event_loop()
        thought = self.node.active_thoughts.get(task_id)
        if not thought:
            return "Task not found"
        return loop.run_until_complete(self.node.query(thought))

    async def query(self, thought: Thought) -> str:
        """Asynchronous query."""
        return await self.node.query(thought)

    async def emergency_stop(self, reason: str, signature: str) -> dict:
        """Article 3: Emergency stop protocol."""
        # In a real implementation, we would verify the signature here
        success = self.kernel.emergency_stop(reason)
        return {"success": success, "reason": reason}

def connect(connection_string: str = "global") -> PleromaSDK:
    """Connect to the Pleroma."""
    return PleromaSDK(node_id=connection_string)

# Re-exporting kernel types for ease of use
from pleroma_kernel import Hyperbolic3 as H3_type, Torus2 as T2_type, Quantum as Quantum_type, Thought as Thought_type
