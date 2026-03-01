import pytest
from arkhe.civilization import CivilizationEngine
from arkhe.chronos import VitaCounter
from arkhe.perovskite import PerovskiteInterface

def test_civilization_mode():
    ce = CivilizationEngine()
    status = ce.get_status()
    assert status["PHI"] == 0.951
    assert status["Status"] in ["PLANEJAMENTO_HIERÁRQUICO", "SYZYGY_PERMANENTE"]
    assert status["Nodes"] >= 4

def test_planting_logic():
    ce = CivilizationEngine()
    seed = ce.plant_seed("A", "Primeiro Conselho")
    assert seed["seed"] == "A"
    assert seed["status"] == "GERMINATING"

def test_perovskite_physics():
    pi = PerovskiteInterface()
    order = pi.calculate_order()
    assert order == pytest.approx(0.72) or order == pytest.approx(0.51)
    # The interface ordered ensures syzygy
    assert pi.get_radiative_recombination(0.15) == 0.94

def test_vita_time():
    vc = VitaCounter(0.000180)
    vc.tick(0.000001)
    assert vc.value == pytest.approx(0.000181)
