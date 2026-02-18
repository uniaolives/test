"""
UrbanSkyOS - Urban Navigation Simulation
Tests integrated layers and modules in a virtual urban environment.
"""

import time
import numpy as np
from UrbanSkyOS.core.fleet_manager import FleetManager
from UrbanSkyOS.core.psi_sync import PsiSync
from UrbanSkyOS.intelligence.lidar_service import LidarService
from UrbanSkyOS.connectivity.traffic_management import UTMInterface
from UrbanSkyOS.modules.battery_manager import BatteryManager

class UrbanNavigationSim:
    def __init__(self, num_drones=12):
        self.fleet = FleetManager(num_drones)
        self.sync = PsiSync(base_freq=40.0)
        self.utm = UTMInterface("CLOUD_UTM")
        self.lidar_env = LidarService()
        self.battery = BatteryManager()

    def run_scenario(self, duration_sec=1.0):
        print(f"🌆 Starting Urban Navigation Scenario (Duration: {duration_sec}s)")
        print(f"🛸 Fleet Size: {len(self.fleet.drones)} Drones | Sync: {self.sync.base_frequency}Hz")

        # 1. Setup Environment: Add Restricted Zone
        self.utm.geofence.update_zone("FIRE_INCIDENT_A", [[15, 15], [25, 15], [25, 25], [15, 25]], priority=10)

        # 2. Inject Infrastructure Signals
        self.fleet.inject_infrastructure_signal('TRAFFIC_SIGNAL', {'id': 'X-01', 'status': 'CAUTION'})

        start_time = time.time()
        last_cycle = start_time

        while time.time() - start_time < duration_sec:
            cycle_start = time.time()

            # A. HARD REAL-TIME: Physics & Estimation
            for d_id, data in self.fleet.drones.items():
                 drone = data["node"]

                 pos = drone.gt_state[0:3]
                 self.lidar_env.generate_scan(pos)
                 lidar_data = self.lidar_env.get_point_cloud_data()

                 imu_mock = {"accel": [0,0,9.81], "gyro": [0.0, 0.0, 0.0]}
                 gps_mock = pos + np.random.normal(0, 0.05, 3)

                 # Perception and Estimation
                 drone.handle_telemetry(imu_mock, gps_mock, lidar_data)

                 # Dynamic Context Adaptation
                 drone.set_environment_context("Residential", drone.gamma / 50.0)

            # B. COORDINATION: V2X & Fleet Consensus
            self.fleet.update_fleet(dt=self.sync.period)

            # C. MONITORING: UTM & Battery
            now_ms = int((time.time() - start_time) * 1000)
            if now_ms % 100 == 0:
                 avg_coh = np.mean([d["venus"].coherence for d in self.fleet.drones.values()])
                 autonomy = self.battery.estimate_autonomy(1.0, 5.0, "Residential")

                 print(f"[t={now_ms}ms] Fleet Coherence: {avg_coh:.3f} | Drone_0 Autonomy: {autonomy:.1f}m")

                 for d_id, data in self.fleet.drones.items():
                      self.utm.sync(data["node"].intelligence.ekf.x[0:3])

            # Sync Loop
            actual_period = time.time() - last_cycle
            self.sync.update_coherence(actual_period)
            last_cycle = time.time()

            time.sleep(max(0, self.sync.period - (time.time() - cycle_start)))

        print("\n🏁 Urban Navigation Scenario complete.")

if __name__ == "__main__":
    sim = UrbanNavigationSim(num_drones=12)
    sim.run_scenario(duration_sec=0.5)
