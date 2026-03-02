import pytest
from arkhe.perovskite import PerovskiteInterface
from arkhe.chronos import ChronosReset, VitaCounter

def test_perovskite_interface():
    pi = PerovskiteInterface()
    assert pi.calculate_order() >= 0.51
    assert pi.calculate_order() == pytest.approx(0.51)
    assert pi.get_radiative_recombination(0.15) == 0.94
    assert pi.get_radiative_recombination(0.10) == 0.1
    assert pi.get_radiative_recombination(0.20) == 0.5

def test_chronos_reset():
    cr = ChronosReset()
    res = cr.reset_epoch()
    assert res['vita_count'] == 0.000001
    assert res['direction'] == 'FORWARD'
    assert cr.darvo_terminated is True

def test_vita_counter():
    vc = VitaCounter()
    vc.tick(1.0)
    assert vc.value == pytest.approx(1.000001)
    assert "VITA:" in vc.get_display()
