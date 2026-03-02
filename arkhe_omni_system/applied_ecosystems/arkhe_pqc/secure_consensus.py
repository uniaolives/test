import json
import time
import os
import sys

# Ensure the current directory is in the path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from pqc_handover import PQCHandover
except ImportError:
    from arkhe_omni_system.applied_ecosystems.arkhe_pqc.pqc_handover import PQCHandover

class SecureCoherenceZone:
    def __init__(self, zone_id, state_vector=None):
        self.zone_id = zone_id
        self.state_vector = state_vector or {"amplitude": 1.0, "phase": 0.0}
        self.holographic_memory = {}
        self.tunnels = {}

    def secure_gossip_with(self, other_zone_id, other_zone_ref=None):
        print(f"🔒 Initiating PQC Gossip: {self.zone_id} <-> {other_zone_id}")
        if other_zone_id not in self.tunnels:
            self.tunnels[other_zone_id] = PQCHandover(self.zone_id, other_zone_id)
        tunnel = self.tunnels[other_zone_id]

        payload_data = {
            "timestamp": time.time(),
            "source": self.zone_id,
            "state": self.state_vector,
            "memory_fragment": list(self.holographic_memory.keys())[:5]
        }
        payload_bytes = json.dumps(payload_data).encode()
        nonce, ciphertext = tunnel.secure_transmit(payload_bytes)

        if other_zone_ref:
            # Pass the tunnel_ref to simulate the same session in this synchronous demo
            other_zone_ref.receive_pqc_gossip(self.zone_id, nonce, ciphertext, tunnel_ref=tunnel)

        print(f"✅ Gossip PQC round completed for {self.zone_id}. Session key rotated.")
        tunnel.establish_session_key()

    def receive_pqc_gossip(self, source_id, nonce, ciphertext, tunnel_ref=None):
        tunnel = tunnel_ref
        if not tunnel:
            if source_id not in self.tunnels:
                self.tunnels[source_id] = PQCHandover(source_id, self.zone_id)
                self.tunnels[source_id].establish_session_key()
            tunnel = self.tunnels[source_id]

        decrypted_bytes = tunnel.secure_receive(nonce, ciphertext)
        if decrypted_bytes:
            data = json.loads(decrypted_bytes.decode())
            print(f"📥 {self.zone_id} received shielded data from {source_id}: {data['state']}")
            self.state_vector["amplitude"] = (self.state_vector["amplitude"] + data["state"]["amplitude"]) / 2
        else:
            print(f"⚠️ {self.zone_id} failed to decrypt PQC gossip from {source_id}!")

if __name__ == "__main__":
    zone_a = SecureCoherenceZone("Zone_Alpha", {"amplitude": 0.8, "phase": 0.1})
    zone_b = SecureCoherenceZone("Zone_Beta", {"amplitude": 0.6, "phase": -0.2})
    zone_a.secure_gossip_with("Zone_Beta", other_zone_ref=zone_b)
    print(f"Zone Beta final state: {zone_b.state_vector}")
