"""
UrbanSkyOS Venus Protocol (Refined)
Inter-drone communication layer based on Arkheto parameters.
Enhanced to process V2X infrastructure signals (Emergency Priority, Traffic Signals).
"""

import numpy as np
import time
import json
from collections import deque

class VenusProtocol:
    def __init__(self, drone_id):
        self.drone_id = drone_id

        # Arkheto Parameters
        self.coherence = 0.91
        self.error = 0.095
        self.phase = 0.33

        self.other_drones = {} # id -> data
        self.no_fly_zones = []
        self.infrastructure_signals = {} # type -> data

    def update_pose(self, pos, vel):
        """Broadcasts own state."""
        self.current_pose = np.array(pos)
        self.current_vel = np.array(vel)
        return {
            "drone_id": self.drone_id,
            "pose": pos.tolist() if isinstance(pos, np.ndarray) else pos,
            "vel": vel.tolist() if isinstance(vel, np.ndarray) else vel,
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

    def on_infrastructure_signal(self, signal_type, signal_data):
        """
        Processes V2X infrastructure signals.
        signal_type: 'EMERGENCY_VEHICLE', 'TRAFFIC_SIGNAL', 'UTM_GEO_UPDATE'
        """
        print(f"📡 V2X [Infrastructure]: Received {signal_type} signal.")
        self.infrastructure_signals[signal_type] = {
            "data": signal_data,
            "timestamp": time.time()
        }

        if signal_type == 'EMERGENCY_VEHICLE':
             print(f"   🚨 EMERGENCY PRIORITY: Clearing path for vehicle {signal_data['id']}.")
             # Reduce coherence temporarily to force yielding if necessary
             self.coherence *= 0.8

        elif signal_type == 'TRAFFIC_SIGNAL':
             print(f"   🚥 TRAFFIC SIGNAL: Intersection status is {signal_data['status']}.")

    def check_conflicts(self):
        """
        Detects conflicts using trajectory extrapolation and Arkheto coherence.
        Now considers infrastructure signals.
        """
        resolutions = []
        my_pos = self.current_pose
        my_vel = self.current_vel

        # 2 seconds lookahead
        my_future = my_pos + my_vel * 2.0

        # Check against emergency vehicle proximity if signal active
        if 'EMERGENCY_VEHICLE' in self.infrastructure_signals:
             sig = self.infrastructure_signals['EMERGENCY_VEHICLE']
             if time.time() - sig['timestamp'] < 10.0:
                  ev_pos = np.array(sig['data']['position'])
                  if np.linalg.norm(my_pos - ev_pos) < 50.0:
                       print(f"   ⚠️ Emergency vehicle nearby! Yielding immediately.")
                       return [{"peer_id": "EV", "action": "YIELD", "reason": "Emergency Priority"}]

        for peer_id, data in self.other_drones.items():
            if time.time() - data['timestamp'] > 5.0: continue

            peer_future = data['position'] + data['velocity'] * 2.0
            dist = np.linalg.norm(my_future - peer_future)

            if dist < 10.0:
                peer_coherence = data['arkheto']['coherence']

                # Check for emergency intent in peer
                peer_priority = data['arkheto'].get('priority', 1)

                if self.coherence >= peer_coherence and peer_priority <= 1:
                    res = {"peer_id": peer_id, "action": "MAINTAIN", "reason": "Higher Coherence"}
                else:
                    res = {"peer_id": peer_id, "action": "YIELD", "reason": "Priority Arbitration"}

                resolutions.append(res)

        return resolutions

if __name__ == "__main__":
    v = VenusProtocol("SKY-01")
    v.update_pose([0,0,10], [5,0,0])
    v.on_infrastructure_signal('EMERGENCY_VEHICLE', {'id': 'AMB-42', 'position': [10, 0, 0]})
    v.check_conflicts()
