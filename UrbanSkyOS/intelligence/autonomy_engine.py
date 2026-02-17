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

        # Simple quat prediction omitted for brevity in this simulation

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

class AutonomyEngine:
    def __init__(self, drone_id="URBAN_001"):
        self.drone_id = drone_id
        self.ekf = UrbanEKF()
        self.kphi = KernelPhiLayer()
        self.state_history = deque(maxlen=100)
        self.kernel_state = np.zeros(64)
        self.lidar_quality = 1.0

    def process_telemetry(self, imu, gps=None, lidar_data=None):
        dt = 0.01
        self.ekf.predict_drone(imu, dt)

        if gps is not None:
            self.ekf.update_gps(gps)

        if lidar_data is not None:
            self._update_kernel_from_lidar(lidar_data)

        self.state_history.append(self.ekf.x[0:3].copy())
        return self.ekf.x

    def _update_kernel_from_lidar(self, lidar_data):
        points = np.array(lidar_data.get('points', []))
        if len(points) > 5:
            centroid = np.mean(points, axis=0)
            features = np.zeros(64)
            features[0:3] = centroid / 100.0
            self.kernel_state = features
            self.lidar_quality = min(1.0, len(points) / 1000.0)
