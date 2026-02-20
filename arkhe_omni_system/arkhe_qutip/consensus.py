# arkhe_qutip/network.py
import asyncio
import hashlib
import time
import random
import numpy as np
from typing import List, Dict, Any, Optional
from .fpga import FPGAQubitEmulator, ArkheFPGAMiner
from .acoustic_time_crystal import AcousticTimeCrystal

class ArkheNetworkNode:
    """
    Represents a geographically distributed node in the Arkhe(N) network.
    Handles QCKD and PoC consensus.
    """
    def __init__(self, node_id: str, location: str):
        self.node_id = node_id
        self.location = location
        self.miner = ArkheFPGAMiner(node_id=node_id)
        self.atc = AcousticTimeCrystal()
        self.qckd_keys: Dict[str, str] = {}
        self.peers: List['ArkheNetworkNode'] = []
        self.blockchain: List[Dict[str, Any]] = []

    def connect(self, peer: 'ArkheNetworkNode'):
        if peer not in self.peers:
            self.peers.append(peer)
            peer.peers.append(self)

    async def qckd_handshake(self, partner_id: str):
        """
        Simulates Quantum Coherence Key Distribution.
        Establishes a shared 'Ethical Context' secret.
        """
        # Sifting simulation
        bits = [random.randint(0, 1) for _ in range(256)]
        bases = [random.choice(['X', 'Z']) for _ in range(256)]

        # In a real system, bases would be exchanged and bits discarded if bases mismatch
        shared_bits = "".join(map(str, bits[:128]))
        shared_key = hashlib.sha256(shared_bits.encode()).hexdigest()

        self.qckd_keys[partner_id] = shared_key
        print(f"[{self.node_id}] QCKD with {partner_id} SUCCESS. Key: {shared_key[:8]}...")
        return shared_key

    async def run_mining_round(self, block_header: Dict[str, Any], target_phi: float):
        """Executes a mining round for the given block."""
        print(f"[{self.node_id}] Mining block at {self.location}...")
        result = self.miner.mine(block_header, target_phi=target_phi)
        return result

    async def run_pocp_round(self, block_header: Dict[str, Any], target_phi: float):
        """
        Executes a Proof of Coherence Physical (PoCP) round.
        Uses the Acoustic Time Crystal as the entropy source.
        """
        print(f"[{self.node_id}] Performing PoCP mining with ATC at {self.location}...")
        start_time = time.time()

        # Simulate ATC stability for a period
        for _ in range(100):
            self.atc.step(dt=0.01)

        phi = self.atc.calculate_phi()
        if phi > target_phi:
            return {
                'node_id': self.node_id,
                'phi_achieved': phi,
                'proof_type': 'PoCP',
                'atc_status': self.atc.get_status(),
                'timestamp': time.time()
            }
        return None

class DistributedPoCConsensus:
    """
    Orchestrates the Proof-of-Coherence consensus across multiple nodes.
    """
    def __init__(self, nodes: List[ArkheNetworkNode]):
        self.nodes = nodes
        self.target_phi = 0.847
        self.block_time_target = 10.0 # seconds for simulation

    async def start_cycle(self, mode: str = 'PoC'):
        """Simulates one consensus cycle (finding the next block)."""
        header = {
            'prev_hash': '0'*64,
            'timestamp': time.time(),
            'merkle_root': hashlib.sha256(b"tx_data").hexdigest()
        }

        if mode == 'PoCP':
            tasks = [node.run_pocp_round(header, self.target_phi) for node in self.nodes]
        else:
            tasks = [node.run_mining_round(header, self.target_phi) for node in self.nodes]

        # Wait for the first winner
        done, pending = await asyncio.wait(
            [asyncio.create_task(t) for t in tasks],
            return_when=asyncio.FIRST_COMPLETED
        )

        for task in done:
            winner_result = task.result()
            if winner_result:
                print(f"\n🏆 WINNER: {winner_result['node_id']} found block with Φ={winner_result['phi_achieved']:.4f}")
                # Cancel remaining tasks
                for p in pending:
                    p.cancel()

                # Validation round
                validators = [n for n in self.nodes if n.node_id != winner_result['node_id']]
                print(f"📡 Validating block with {len(validators)} peers...")

                # In Arkhe(N), validation involves checking the coherence signature
                return winner_result

        return None
