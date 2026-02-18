"""
UrbanSkyOS Data Exporter
Provides state data for monitoring and visualization components.
"""

import json
import os
import time
import numpy as np

class DataExporter:
    def __init__(self, output_file="fleet_state.json"):
        self.output_file = output_file
        self.last_export = 0

    def export_fleet_state(self, fleet_manager):
        current_time = time.time()

        fleet_data = []
        coherences = []

        for d_id, data in fleet_manager.drones.items():
            pose = data["node"].intelligence.ekf.x[0:3].tolist()
            vel = data["node"].intelligence.ekf.x[3:6].tolist()
            coherence = data["venus"].coherence

            coherences.append(coherence)

            drone_entry = {
                "id": d_id,
                "position": pose,
                "velocity": vel,
                "coherence": coherence,
                "uncertainty": np.trace(data["node"].intelligence.ekf.P[0:3, 0:3]),
                "gamma": data["node"].gamma
            }
            fleet_data.append(drone_entry)

        collective_state = {
            "timestamp": current_time,
            "avg_coherence": np.mean(coherences) if coherences else 0,
            "fleet_size": len(fleet_data),
            "waypoint": fleet_manager.fleet_waypoint.tolist()
        }

        full_state = {
            "collective": collective_state,
            "drones": fleet_data
        }

        try:
            with open(self.output_file, 'w') as f:
                json.dump(full_state, f)
        except Exception as e:
            print(f"Export error: {e}")

        self.last_export = current_time
        return full_state
