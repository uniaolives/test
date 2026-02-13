# tests/test_governance.py
import pytest
from arkhe.geodesic import VirologicalGovernance, MaturityStatus, LatentFocus

def test_monolayer_capacity():
    stones = [
        LatentFocus(1, "s1", 10.0, 0.07, 0.9, True, 0.05),
        LatentFocus(2, "s2", 10.0, 0.07, 0.9, True, 0.05),
    ]
    gov = VirologicalGovernance(
        maturity_status=MaturityStatus.MATURE,
        latent_stones=stones,
        max_safe_occupancy=0.25
    )

    assert gov.calculate_current_occupancy() == 0.10
    assert gov.check_capacity(0.10) == True
    assert gov.check_capacity(0.20) == False

def test_full_capacity():
    stones = [LatentFocus(i, "s", 10.0, 0.07, 0.9, True, 0.05) for i in range(5)]
    gov = VirologicalGovernance(
        maturity_status=MaturityStatus.MATURE,
        latent_stones=stones,
        max_safe_occupancy=0.25
    )
    # 5 * 0.05 = 0.25
    assert gov.calculate_current_occupancy() == 0.25
    assert gov.check_capacity(0.01) == False
