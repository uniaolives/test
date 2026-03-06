import pytest
from gateway.app.geoloc.poloc import BftPoLoc
from gateway.app.geoloc.utils import haversine

def test_haversine():
    # Distance from London (51.5074, -0.1278) to Paris (48.8566, 2.3522) is ~344km
    dist = haversine(51.5074, -0.1278, 48.8566, 2.3522)
    assert abs(dist - 344) < 5

def test_poloc_honest():
    verifier = BftPoLoc(beta=0.2)
    claimed_lat, claimed_lon = 0, 0
    # Honest measurements (RTT matches distance)
    # 500km = 5ms RTT
    measurements = [
        {'lat': 0, 'lon': 4.5, 'rtt': 10}, # ~500km
        {'lat': 4.5, 'lon': 0, 'rtt': 10},
        {'lat': 0, 'lon': -4.5, 'rtt': 10},
        {'lat': -4.5, 'lon': 0, 'rtt': 10},
        {'lat': 3.18, 'lon': 3.18, 'rtt': 10},
    ]
    result = verifier.verify(claimed_lat, claimed_lon, measurements, threshold_km=100)
    assert result["is_valid"]
    assert result["R_star"] < 100

def test_poloc_byzantine():
    # 2 out of 5 are Byzantine (40%) -> beta=0.2 only tolerates 1/5.
    # If beta=0.4, it should tolerate 2/5.
    verifier = BftPoLoc(beta=0.4)
    claimed_lat, claimed_lon = 0, 0

    measurements = [
        {'lat': 0, 'lon': 4.5, 'rtt': 10}, # Honest (~500km)
        {'lat': 4.5, 'lon': 0, 'rtt': 10}, # Honest
        {'lat': 0, 'lon': -4.5, 'rtt': 10}, # Honest
        {'lat': 10, 'lon': 10, 'rtt': 10},  # Byzantine (Far away but low RTT)
        {'lat': -10, 'lon': -10, 'rtt': 10}, # Byzantine
    ]

    # With beta=0.4, it picks the 3rd smallest residual (honest ones)
    result = verifier.verify(claimed_lat, claimed_lon, measurements, threshold_km=100)
    assert result["is_valid"]

    # With beta=0.2, it picks the 4th smallest residual (one Byzantine)
    verifier_low_beta = BftPoLoc(beta=0.2)
    result_low = verifier_low_beta.verify(claimed_lat, claimed_lon, measurements, threshold_km=100)
    assert not result_low["is_valid"]
