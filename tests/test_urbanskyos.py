import pytest
import numpy as np
from UrbanSkyOS.core.fleet_simulation import FleetSimulation
from UrbanSkyOS.core.kernel_phi import KernelPhiLayer
from UrbanSkyOS.intelligence.autonomy_engine import AutonomyEngine
from UrbanSkyOS.modules.noise_reduction import NoiseReduction

def test_fleet_simulation_consensus():
    fs = FleetSimulation(num_drones=3)
    initial_vel = fs.drones["drone_0"].gt_state[3:6].copy()
    fs.update_fleet(dt=0.1)
    # Velocity should change toward celestial target
    assert not np.array_equal(fs.drones["drone_0"].gt_state[3:6], initial_vel)

def test_adaptive_gamma():
    kphi = KernelPhiLayer(gamma=1.0)
    new_gamma = kphi.adapt_gamma(coherence=0.4, safety_metric=0.2)
    assert new_gamma > 1.0

def test_noise_reduction_optimization():
    nr = NoiseReduction()
    rpms = nr.optimize_rpms(500, 0, 0, 0)
    assert len(rpms) == 4
    assert np.all(rpms >= 100)

def test_ekf_multi_sensor():
    ae = AutonomyEngine()
    imu = {'accel': [1, 0, 9.81], 'gyro': [0, 0, 0]}
    gps = [1, 1, 10]
    lidar = {'points': [[1,1,1], [2,2,2], [3,3,3], [4,4,4], [5,5,5], [6,6,6]]}
    state = ae.process_telemetry(imu, gps, lidar)
    assert abs(state[0]) > 0
