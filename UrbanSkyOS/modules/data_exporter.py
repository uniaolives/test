"""
UrbanSkyOS Data Exporter
Provides state data for monitoring and visualization components (e.g., D3.js).
Includes RKHS metrics: coherence, kernel density, and uncertainty.
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
        """
        Exports current fleet positions, trajectories, and collective RKHS state.
        """
        current_time = time.time()

        fleet_data = []
        positions = []
        coherences = []

        # Accessing drones via FleetManager or similar structure
        # Since different scenarios might use different managers,
        # we assume a dict-like drones attribute.
        drones_dict = getattr(fleet_manager, 'drones', {})

        for d_id, data in drones_dict.items():
            # In some simulations data is a dict, in others it might be different
            # Handling both cases
            if isinstance(data, dict):
                node = data["node"]
                venus = data["venus"]
            else:
                node = data
                venus = getattr(node, 'venus', None)

            pose = node.intelligence.ekf.x[0:3].tolist()
            vel = node.intelligence.ekf.x[3:6].tolist()
            coherence = getattr(venus, 'coherence', 1.0)

            positions.append(pose)
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
                "uncertainty": float(np.trace(node.intelligence.ekf.P[0:3, 0:3])),
                "gamma": getattr(node, 'gamma', 50.0)
            }
            fleet_data.append(drone_entry)

        # Collective state (RKHS)
        avg_coh = np.mean(coherences) if coherences else 0

        # Kernel Density simulation (Simplified)
        # In a real RKHS viz, this would be the sum of kernels
        kernel_density = []
        if positions:
            # Sample density at a few points
            kernel_density = [
                {"pos": [0,0,0], "density": avg_coh},
                {"pos": [50,50,0], "density": avg_coh * 0.5}
            ]

        collective_state = {
            "timestamp": current_time,
            "avg_coherence": avg_coh,
            "fleet_size": len(fleet_data),
            "waypoint": getattr(fleet_manager, 'fleet_waypoint', [0,0,0]).tolist() if hasattr(fleet_manager, 'fleet_waypoint') else [0,0,0],
            "kernel_density": kernel_density
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

        # Write to file for D3.js visualizers
        try:
            with open(self.output_file, 'w') as f:
                json.dump(full_state, f, indent=2)
        try:
            with open(self.output_file, 'w') as f:
                json.dump(full_state, f)
        except Exception as e:
            print(f"Export error: {e}")

        self.last_export = current_time
        return full_state

if __name__ == "__main__":
    # Test stub
    pass
