"""
UrbanSkyOS - Main Integration Script (Refined)
Orchestrates refined services and provides data for:
- PointCloudViewer (Lidar)
- SensorTelemetryPanel (Lidar/FC)
- SignalScope (Lidar Intensity)
"""

import time
import numpy as np
from UrbanSkyOS.core.drone_node import DroneNode
from UrbanSkyOS.intelligence.autonomy_engine import AutonomyEngine
from UrbanSkyOS.intelligence.lidar_service import LidarService
from UrbanSkyOS.intelligence.venus_protocol import VenusProtocol
from UrbanSkyOS.connectivity.traffic_management import UTMInterface
from UrbanSkyOS.modules.privacy_filter import PrivacyBlur

class UrbanSkyOS:
    def __init__(self, drone_id="URBAN_SKY_01"):
        self.drone_id = drone_id
        self.drone = DroneNode(drone_id)
        self.intelligence = AutonomyEngine(drone_id)
        self.lidar = LidarService()
        self.venus = VenusProtocol(drone_id)
        self.utm = UTMInterface(drone_id)
        self.privacy = PrivacyBlur()

        self.last_handover_time = 0

    def run_simulation(self, duration_sec=1.0):
        print(f"🚀 UrbanSkyOS Integration Node active: {self.drone_id}")

        # Simulate initial UTM zone update
        self.utm.geofence.update_zone("CONCERT_2026", [[40, 50], [45, 50], [45, 55]], "2026-02-18T18:00", "2026-02-18T23:00", reason="event")

        start_time = time.time()
        while time.time() - start_time < duration_sec:
            # 1. HARD REAL-TIME: Control Loop (1kHz)
            # Motor commands are ANR-optimized
            motor_cmds = self.drone.control_loop(dt=0.001)

            # 2. SOFT REAL-TIME: Federation Handover & UI Component Provisioning (10Hz)
            now_ms = int((time.time() - start_time) * 1000)
            if now_ms - self.last_handover_time >= 100:
                print(f"\n[t={now_ms}ms] --- Handover & Component Sync ---")

                # A. Intelligence & Navigation
                self.intelligence.run_cycle()
                current_pose = self.intelligence.ekf.x[0:3]
                current_vel = self.intelligence.ekf.x[3:6]

                # B. Lidar Service (Provisioning components)
                self.lidar.generate_scan(current_pose)

                # PointCloudViewer Sync
                pc_data = self.lidar.get_point_cloud_data()
                # print(f"   🖥️ PointCloudViewer: Received {len(pc_data['points'])} points.")

                # SensorTelemetryPanel Sync
                lidar_telemetry = self.lidar.get_telemetry()
                fc_telemetry = {"motors": motor_cmds.tolist(), "pose": current_pose.tolist()}
                # print(f"   📊 SensorTelemetryPanel: Status={lidar_telemetry['status']}, Points={lidar_telemetry['points_count']}")

                # SignalScope Sync
                signal_data = self.lidar.get_signal_scope_data()
                # print(f"   📉 SignalScope: Received {len(signal_data)} intensity samples.")

                # C. Venus Protocol Sync
                broadcast = self.venus.update_pose(current_pose, current_vel)
                # Mocking a peer conflict
                self.venus.on_peer_broadcast({
                    "drone_id": "SKY-VULCAN-22", "pose": current_pose + [2, 0, 0],
                    "vel": [-1, 0, 0], "arkheto": {"coherence": 0.85}
                })
                self.venus.check_conflicts()

                # D. UTM Sync
                self.utm.sync(current_pose)

                self.last_handover_time = now_ms

            time.sleep(0.001) # Approx loop speed

        print("\n🏁 Simulation complete.")

if __name__ == "__main__":
    os = UrbanSkyOS()
    os.run_simulation(duration_sec=0.5)
