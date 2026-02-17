"""
UrbanSkyOS Venus Protocol (Refined)
Inter-drone communication layer based on Arkheto parameters (Ψ-layer).
Handles adaptive geofencing and conflict resolution via trajectory extrapolation.
"""

import numpy as np
import time
import json
from collections import deque

class VenusProtocol:
    def __init__(self, drone_id):
        self.drone_id = drone_id

        # Arkheto Parameters (Ψ-layer)
        self.coherence = 0.91
        self.error = 0.095
        self.phase = 0.33

        self.other_drones = {} # id -> {'position': np.array, 'timestamp': float, 'velocity': np.array}
        self.no_fly_zones = []

    def update_pose(self, pos, vel):
        """Broadcasts own state."""
        self.current_pose = np.array(pos)
        self.current_vel = np.array(vel)
        return {
            "drone_id": self.drone_id,
            "pose": pos,
            "vel": vel,
            "arkheto": {"coherence": self.coherence, "error": self.error}
        }

    def on_peer_broadcast(self, peer_data):
        """Receives state from peer."""
        peer_id = peer_data["drone_id"]
        if peer_id == self.drone_id: return

        self.other_drones[peer_id] = {
            "position": np.array(peer_data["pose"]),
            "velocity": np.array(peer_data["vel"]),
            "arkheto": peer_data["arkheto"],
            "timestamp": time.time()
        }

    def check_conflicts(self):
        """
        Detects conflicts using trajectory extrapolation and Arkheto coherence.
        Returns a list of resolutions.
        """
        resolutions = []
        my_pos = self.current_pose
        my_vel = self.current_vel

        # 2 seconds lookahead
        my_future = my_pos + my_vel * 2.0

        for peer_id, data in self.other_drones.items():
            if time.time() - data['timestamp'] > 5.0: continue

            peer_future = data['position'] + data['velocity'] * 2.0

            # Simple RBF-like similarity check for futures
            dist = np.linalg.norm(my_future - peer_future)

            if dist < 10.0: # Conflict threshold
                print(f"📡 Venus: Potential conflict with {peer_id} detected (dist_future={dist:.2f}m).")

                # Resolve using Coherence (Arkheto)
                peer_coherence = data['arkheto']['coherence']

                if self.coherence >= peer_coherence:
                    # Higher coherence (stability) maintains path
                    res = {"peer_id": peer_id, "action": "MAINTAIN", "reason": "Higher Coherence"}
                else:
                    # Lower coherence yields to stable nodes
                    res = {"peer_id": peer_id, "action": "YIELD", "reason": "Lower Coherence"}

                resolutions.append(res)
                print(f"   ✅ Decision for {self.drone_id}: {res['action']} ({res['reason']})")

        return resolutions

    def process_utm_update(self, zone_json):
        """Adaptive geofencing update."""
        zone = json.loads(zone_json)
        self.no_fly_zones.append(zone)
        print(f"🛑 Venus: Adaptive geofence added for {self.drone_id}: {zone['id']}")

if __name__ == "__main__":
    v = VenusProtocol("SKY-01")
    v.update_pose([0,0,10], [5,0,0])
    v.on_peer_broadcast({
        "drone_id": "SKY-02",
        "pose": [15, 0, 10],
        "vel": [-1, 0, 0],
        "arkheto": {"coherence": 0.8}
    })
    v.check_conflicts()
