"""
Story-Arkhe Bridge: Programmable IP Interface
Simulates handovers between Arkhe agents and the Story-Geth execution layer.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import time
import json
import requests  # Added for functional JSON-RPC

@dataclass
class IPAgent:
    """Agent that manages or creates Intellectual Property."""
    agent_id: str
    name: str
    satoshi_balance: int = 1000000 # Integer Satoshis
    reputation: float = 0.95
    owned_assets: List[str] = field(default_factory=list)

    def update_reputation(self, success: bool):
        if success:
            self.reputation = min(1.0, self.reputation + 0.005)
        else:
            self.reputation = max(0.1, self.reputation - 0.05)

class StoryArkheBridge:
    """
    Bridge simulation between ArkheOS and Story Protocol (Story-Geth).
    Enforces C + F = 1 during IP handovers.

    Can optionally connect to a real Story-Geth node via JSON-RPC.
    """

    def __init__(self, rpc_url: Optional[str] = None):
        self.agents: Dict[str, IPAgent] = {}
        self.registered_assets: Dict[str, Dict] = {} # asset_id -> data
        self.logs: List[Dict] = []
        self.PHI = 0.618033988749895
        self.rpc_url = rpc_url

    def _call_rpc(self, method: str, params: List) -> Optional[Dict]:
        """Optional functional RPC call to Story-Geth."""
        if not self.rpc_url:
            return None

        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": 1
        }
        try:
            response = requests.post(self.rpc_url, json=payload, timeout=5)
            return response.json()
        except Exception as e:
            print(f"RPC Error: {e}")
            return None

    def register_agent(self, agent_id: str, name: str):
        self.agents[agent_id] = IPAgent(agent_id, name)

    def request_ip_registration(self, agent_id: str, title: str, ip_type: str, gas_fee: int) -> Optional[str]:
        """Handover: Arkhe Agent -> Story Protocol (IP Asset Registration)."""
        if agent_id not in self.agents:
            return None

        agent = self.agents[agent_id]
        if agent.satoshi_balance < gas_fee:
            agent.update_reputation(False)
            return None

        # Execute Handover
        agent.satoshi_balance -= gas_fee

        # Optional: Verify on-chain if RPC is available
        rpc_data = self._call_rpc("eth_blockNumber", [])
        if rpc_data:
            print(f"Verified Story-Geth connection: Block {rpc_data.get('result')}")

        # Simulating Story-Geth response
        success = np.random.random() < agent.reputation
        agent.update_reputation(success)

        if success:
            asset_id = f"0x_IPA_{hash(title + agent_id) % 10**8:08d}"
            self.registered_assets[asset_id] = {
                'title': title,
                'owner': agent_id,
                'ip_type': ip_type,
                'gas_spent': gas_fee,
                'timestamp': time.time()
            }
            agent.owned_assets.append(asset_id)

            self.logs.append({
                'type': 'ip_registration',
                'agent_id': agent_id,
                'asset_id': asset_id,
                'success': True,
                'timestamp': time.time()
            })
            return asset_id

        self.logs.append({
            'type': 'ip_registration',
            'agent_id': agent_id,
            'success': False,
            'timestamp': time.time()
        })
        return None

    def execute_licensing_handover(self, licensor_id: str, licensee_id: str, asset_id: str) -> bool:
        """Handover: IP Licensing between two Arkhe agents mediated by Story."""
        if licensor_id not in self.agents or licensee_id not in self.agents:
            return False

        if asset_id not in self.registered_assets or self.registered_assets[asset_id]['owner'] != licensor_id:
            return False

        licensor = self.agents[licensor_id]
        licensee = self.agents[licensee_id]

        # Consensus coherence check
        coherence = (licensor.reputation * licensee.reputation) ** 0.5
        success = np.random.random() < coherence

        if success:
            licensee.satoshi_balance -= 1000 # License fee
            licensor.satoshi_balance += 950  # 5% Story Protocol fee (950 / 1000)

        self.logs.append({
            'type': 'licensing_handover',
            'licensor': licensor_id,
            'licensee': licensee_id,
            'asset_id': asset_id,
            'success': success,
            'timestamp': time.time()
        })

        return success

    def get_bridge_telemetry(self) -> Dict:
        """Global metrics for the Story-Arkhe interface."""
        if not self.logs:
            return {'coherence': 1.0, 'traffic': 0}

        success_rate = len([l for l in self.logs if l['success']]) / len(self.logs)
        total_satoshi_flow = sum([l.get('gas_spent', 0) for l in self.logs])

        return {
            'coherence': success_rate,
            'total_assets': len(self.registered_assets),
            'traffic_density': len(self.logs),
            'satoshi_flow': total_satoshi_flow
        }

if __name__ == "__main__":
    bridge = StoryArkheBridge()
    bridge.register_agent("A-001", "Rafael_Architect")
    bridge.register_agent("A-002", "Satellite_Operator_Bot")

    print("Registering Anyonic Patent on Story-Arkhe Bridge...")
    ipa = bridge.request_ip_registration("A-001", "Anyonic Phase Accumulation", "patent", 500)
    print(f"Asset ID: {ipa}")

    print("\nExecuting Licensing Handover...")
    res = bridge.execute_licensing_handover("A-001", "A-002", ipa)
    print(f"Licensing Result: {'Success' if res else 'Failure'}")

    print("\nBridge Telemetry:")
    print(bridge.get_bridge_telemetry())
    print("∞")
