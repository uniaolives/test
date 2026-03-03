"""
UrbanSkyOS Fleet Simulation
Manages multiple DroneNode instances and coordinates movement toward celestial targets.
"""

import numpy as np
from typing import Dict, List
from UrbanSkyOS.core.drone_node import DroneNode
from UrbanSkyOS.intelligence.venus_protocol import VenusProtocol

class CelestialTarget:
    def __init__(self, name, ra, dec):
        self.name = name
        self.ra = ra
        self.dec = dec

    def get_direction(self):
        # Simplified celestial to local mapping
        ra_rad = self.ra * 15.0 * np.pi / 180.0
        dec_rad = self.dec * np.pi / 180.0
        return np.array([np.cos(dec_rad)*np.cos(ra_rad), np.cos(dec_rad)*np.sin(ra_rad), np.sin(dec_rad)])

class FleetSimulation:
    def __init__(self, num_drones=12):
        self.num_drones = num_drones
        self.drones: Dict[str, DroneNode] = {}
        self.drone_ids = [f"drone_{i}" for i in range(num_drones)]

        # Targets
        self.targets = {
            'sirius': CelestialTarget('Sirius', 6.75, -16.72),
            'galactic_center': CelestialTarget('Galactic Center', 17.76, -29.00)
        }
        self.current_target = self.targets['sirius']

        # Formation positions
        for i, d_id in enumerate(self.drone_ids):
            angle = 2 * np.pi * i / num_drones
            pos = np.array([30 * np.cos(angle), 30 * np.sin(angle), 50.0])
            node = DroneNode(d_id)
            node.gt_state[0:3] = pos
            self.drones[d_id] = node

    def update_fleet(self, dt=0.01):
        target_dir = self.current_target.get_direction()

        # 1. Update each drone
        states = {}
        for d_id, drone in self.drones.items():
            # Apply consensus pull toward target_dir
            pull = target_dir * 10.0
            drone.gt_state[3:6] = 0.9 * drone.gt_state[3:6] + 0.1 * pull

            # Run control loop
            drone.control_loop(dt)
            states[d_id] = {
                "pose": drone.intelligence.ekf.x[0:3],
                "kernel": drone.intelligence.kernel_state,
                "coherence": drone.intelligence.kphi.uncertainty_quantification(list(drone.intelligence.state_history)[-10:], drone.intelligence.ekf.x[0:3])['coherence_with_data'] if len(drone.intelligence.state_history) > 10 else 1.0
            }

        return states

    def get_fleet_state(self):
        return {d_id: {"pos": d.gt_state[0:3].tolist(), "coh": 0.9} for d_id, d in self.drones.items()}
