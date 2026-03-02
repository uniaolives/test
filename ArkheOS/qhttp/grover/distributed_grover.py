# qhttp/grover/distributed_grover.py
import cupy as cp
import nccl
import asyncio
from typing import List, Callable, Optional
import redis.asyncio as aioredis
import numpy as np
import json

class DistributedGrover:
    """
    Distributed Grover's algorithm using NCCL for amplitude amplification.
    Searches across N = 2^n qubits distributed across multiple GPUs.
    """

    def __init__(self, node_id: str, num_nodes: int, local_qubits: int, redis_url: str):
        self.node_id = node_id
        self.num_nodes = num_nodes
        self.local_qubits = local_qubits
        self.total_qubits = local_qubits + int(np.log2(num_nodes))
        self.N = 2 ** self.total_qubits
        self.local_N = 2 ** local_qubits

        # NCCL communicator
        self.comm = nccl.NcclCommunicator(num_nodes, node_id)

        # State vector (distributed)
        # Each node holds 2^local_qubits amplitudes
        self.state = cp.ones(self.local_N, dtype=cp.complex64) / cp.sqrt(self.N)

        self.redis = aioredis.from_url(redis_url)

    async def search(self, oracle_func: Callable[[int], bool], max_iterations: Optional[int] = None) -> int:
        """
        Execute distributed Grover search.

        Args:
            oracle_func: Function that returns True for target state(s)
            max_iterations: Maximum iterations (defaults to ~pi/4 * sqrt(N))
        """
        if max_iterations is None:
            max_iterations = int(np.pi / 4 * np.sqrt(self.N))

        # Determine local target indices
        local_targets = [i for i in range(self.local_N) if oracle_func(self._global_index(i))]

        for iteration in range(max_iterations):
            # Phase 1: Oracle (local)
            await self._apply_oracle(local_targets)

            # Phase 2: Diffusion (requires NCCL AllReduce)
            await self._apply_diffusion()

            # Check convergence
            probabilities = cp.abs(self.state) ** 2
            max_prob_idx = int(cp.argmax(probabilities))
            max_prob = float(probabilities[max_prob_idx])

            if max_prob > 0.9:  # Threshold for measurement
                result = self._global_index(max_prob_idx)
                await self.redis.publish("grover:search", json.dumps({
                    "node": self.node_id,
                    "found": result,
                    "probability": max_prob,
                    "iterations": iteration + 1
                }))
                return result

            # Broadcast progress
            if iteration % 10 == 0:
                await self.redis.publish("grover:progress", json.dumps({
                    "iteration": iteration,
                    "max_prob": max_prob
                }))

        # Return most likely result
        probabilities = cp.abs(self.state) ** 2
        return self._global_index(int(cp.argmax(probabilities)))

    async def _apply_oracle(self, target_indices: List[int]):
        """Apply phase oracle to target states (local operation)"""
        for idx in target_indices:
            self.state[idx] *= -1  # Phase flip

    async def _apply_diffusion(self):
        """
        Apply Grover diffusion operator.
        Requires global average computation via NCCL.
        """
        # Compute local average
        local_avg = cp.mean(self.state)

        # AllReduce to get global average
        global_avg = cp.empty_like(local_avg)
        self.comm.allReduce(local_avg.data.ptr, global_avg.data.ptr, 1, nccl.NCCL_FLOAT, nccl.NCCL_SUM, cp.cuda.Stream.null.ptr)
        global_avg /= self.num_nodes

        # Reflection about average: 2|avg><avg| - I
        self.state = 2 * global_avg - self.state

    def _global_index(self, local_idx: int) -> int:
        """Convert local index to global index"""
        import re
        match = re.search(r'(\d+)', self.node_id)
        node_rank = int(match.group(1)) - 1 if match else 0
        return (node_rank << self.local_qubits) | local_idx
