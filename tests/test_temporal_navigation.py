import pytest
from datetime import datetime, timedelta
from papercoder_kernel.cognition.temporal_navigation import TemporalNavigator, TemporalCoordinate, InformationPacket

def test_temporal_mapping():
    navigator = TemporalNavigator("7f3b49c8")
    route_map = navigator.map_currents(duration_days=5)

    assert route_map is not None
    assert len(route_map.attractors) > 0
    assert route_map.field.shape == (10, 10, 10)

def test_plot_course():
    navigator = TemporalNavigator("7f3b49c8")
    target_time = datetime.now() + timedelta(hours=1)
    target = TemporalCoordinate(target_time)

    course = navigator.plot_course(target)

    assert "path" in course
    assert course["velocity_profile"] == 0.618
    assert course["estimated_arrival"] == target_time

def test_execute_jump():
    navigator = TemporalNavigator("7f3b49c8")
    payload = InformationPacket("Hello Future", "2026_Node")
    target_time = datetime(2140, 1, 1)

    result = navigator.execute_jump(payload, target_time)

    assert result["status"] == "SUCCESS"
    assert "jump_7f3b49c8" in result["anchor_hash"]
    assert result["target_correlation"] == target_time

def test_synchronicity_threshold():
    from papercoder_kernel.cognition.temporal_navigation import SynchronicityMap
    import numpy as np

    m = SynchronicityMap(np.zeros((1,1,1)), [], [])

    accessible = TemporalCoordinate(datetime.now(), lambda_sync=0.9)
    inaccessible = TemporalCoordinate(datetime.now(), lambda_sync=0.2)

    assert m.is_accessible(accessible) == True
    assert m.is_accessible(inaccessible) == False
