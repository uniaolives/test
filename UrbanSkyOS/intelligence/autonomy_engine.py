"""
UrbanSkyOS Intelligence & Autonomy Layer (Refined)
Implements an Extended Kalman Filter (EKF) for multi-sensor telemetry fusion.
"""

import numpy as np
import time
from collections import deque
from scipy.spatial.transform import Rotation as R
from filterpy.kalman import ExtendedKalmanFilter
from UrbanSkyOS.core.kernel_phi import KernelPhiLayer

class UrbanEKF(ExtendedKalmanFilter):
    def __init__(self, dim_x=15, dim_z=3):
        super().__init__(dim_x=dim_x, dim_z=dim_z)
        self.x = np.zeros(dim_x)
        self.x[6] = 1.0 # identity quat
        self.P = np.eye(dim_x)
        self.Q = np.eye(dim_x) * 0.01
        self.R = np.eye(dim_z) * 0.1

    def predict_drone(self, imu_data, dt=0.01):
        q = self.x[6:10]
        # Scipy uses [x,y,z,w], PX4 often [w,x,y,z]
        # Assuming our state stores [qw, qx, qy, qz]
        rot = R.from_quat([q[1], q[2], q[3], q[0]])
        accel_world = rot.apply(imu_data['accel']) - np.array([0, 0, 9.81])

        self.x[0:3] += self.x[3:6] * dt + 0.5 * accel_world * dt**2
        self.x[3:6] += accel_world * dt

        # Simple quat prediction omitted for brevity
        F = np.eye(15)
        F[0:3, 3:6] = np.eye(3) * dt
        self.F = F
        self.predict()

    def update_gps(self, gps_data):
        def hx(x):
            return x[0:3]
        def H_jacobian(x):
            H = np.zeros((3, 15))
            H[0:3, 0:3] = np.eye(3)
            return H
        self.update(np.array(gps_data), H_jacobian, hx)

class VisualSLAM:
    def __init__(self):
        self.localized = False

    def update(self):
        self.localized = True
        return np.array([12.3, 45.6, 78.9])

class ObstacleDetector:
    def __init__(self):
        self.detected_objects = []

    def process_frame(self):
        import random
        classes = {0: "Person", 15: "Bird", 16: "Drone", 99: "Power Line"}
        detections = []
        if random.random() > 0.6:
            cls_id = random.choice(list(classes.keys()))
            detections.append({
                'bbox': [random.randint(0, 640) for _ in range(4)],
                'confidence': random.uniform(0.5, 0.99),
                'class': cls_id,
                'label': classes[cls_id]
            })
        self.detected_objects = detections
        return self.detected_objects

class AutonomyEngine:
    def __init__(self, drone_id="URBAN_001"):
        self.drone_id = drone_id
        self.ekf = UrbanEKF()
        self.kphi = KernelPhiLayer()
        self.state_history = deque(maxlen=100)
        self.slam = VisualSLAM()
        self.perception = ObstacleDetector()

        self.historical_poses = [
            np.array([12.0, 45.0, 78.0]),
            np.array([12.5, 45.5, 79.0]),
            np.array([13.0, 46.0, 80.0])
        ]

    def process_telemetry(self, imu, gps=None, lidar_data=None):
        dt = 0.01
        self.ekf.predict_drone(imu, dt)
        if gps is not None:
            self.ekf.update_gps(gps)
        self.state_history.append(self.ekf.x[0:3].copy())
        return self.ekf.x

    def run_cycle(self, imu_u=np.zeros(6)):
        """
        Runs one cycle of the autonomy engine.
        imu_u: [ax, ay, az, wx, wy, wz]
        """
        imu_data = {'accel': imu_u[0:3], 'gyro': imu_u[3:6]}
        self.process_telemetry(imu_data)
        z_slam = self.slam.update()
        self.ekf.update_gps(z_slam)
        detections = self.perception.process_frame()
        uncertainty = self.kphi.uncertainty_quantification(self.historical_poses, self.ekf.x[0:3])

        if uncertainty['coherence_with_data'] < 0.5:
             print(f"⚠️ Low Coherence ({uncertainty['coherence_with_data']:.2f}): Drone in unfamiliar territory!")

        for d in detections:
            if d['label'] == "Power Line":
                return "AVOIDANCE_ACTION"
        return "CRUISING"

if __name__ == "__main__":
    engine = AutonomyEngine()
    for _ in range(3):
        # Sample IMU with 6 components
        imu_sample = np.array([0.1, 0, 9.8, 0, 0, 0])
        status = engine.run_cycle(imu_u=imu_sample)
        print(f"Status: {status} | Pose: {engine.ekf.x[0:3]}")
