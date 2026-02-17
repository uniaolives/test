import pytest
import numpy as np
import time
from UrbanSkyOS.core.drone_node import DroneNode
from UrbanSkyOS.intelligence.autonomy_engine import AutonomyEngine
from UrbanSkyOS.intelligence.lidar_service import LidarService
from UrbanSkyOS.intelligence.venus_protocol import VenusProtocol
from UrbanSkyOS.connectivity.traffic_management import UTMInterface
from UrbanSkyOS.modules.noise_reduction import NoiseReduction

def test_lidar_refined_scan():
    ls = LidarService()
    pts = ls.generate_scan((0,0,10))
    # Should generate points if obstacles are within range
    # In random simulation, at least status should be correct
    telemetry = ls.get_telemetry()
    assert "status" in telemetry
    assert "points_count" in telemetry

    intensities = ls.get_signal_scope_data()
    assert len(intensities) == len(pts)

def test_venus_refined_conflict():
    v = VenusProtocol("SKY-01")
    v.update_pose([0,0,10], [5,0,0]) # Moving towards X
    v.on_peer_broadcast({
        "drone_id": "SKY-02", "pose": [15, 0, 10],
        "vel": [-1, 0, 0], "arkheto": {"coherence": 0.8}
    })
    # My coherence is 0.91, peer is 0.8. I should MAINTAIN.
    resolutions = v.check_conflicts()
    assert len(resolutions) > 0
    assert resolutions[0]["action"] == "MAINTAIN"

def test_utm_refined_api():
    utm = UTMInterface()
    res = utm.geofence.update_zone("TEST_Z", [[0,0], [1,0], [1,1]], "now", "later")
    assert res["status"] == "zone added"

    # Position (0,0) is center of first point, should be in zone
    status = utm.sync((0.1, 0.1))
    assert status["in_zone"] is True

def test_anr_least_squares():
    nr = NoiseReduction()
    # For balanced thrust and no torque, motor speeds should be equal
    rpms = nr.optimize_rpms(500, 0, 0, 0)
    assert len(rpms) == 4
    assert np.allclose(rpms[0], rpms[1])

def test_node_refined_control():
    node = DroneNode("TEST")
    # control_loop should return ANR-optimized motor commands
    cmds = node.control_loop(dt=0.001)
    assert len(cmds) == 4
    assert np.all(cmds > 0)
