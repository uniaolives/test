import pytest
from arkhe.civilization import CivilizationEngine
from arkhe.chronos import VitaCounter
from arkhe.perovskite import PerovskiteInterface

def test_civilization_mode():
    ce = CivilizationEngine()
    status = ce.get_status()
    assert status["PHI"] == 0.951
    assert status["Status"] == "PLANEJAMENTO_HIERÁRQUICO"
    assert status["Nodes"] >= 12450

def test_planting_logic():
    ce = CivilizationEngine()
    seed = ce.plant_seed("A", "Primeiro Conselho")
    assert seed["seed"] == "A"
    assert seed["status"] == "GERMINATING"

def test_perovskite_physics():
    pi = PerovskiteInterface()
    assert pi.calculate_order() == pytest.approx(0.72)
    # The interface ordered ensures syzygy
    assert pi.get_radiative_recombination(0.15) == 0.94

def test_vita_time():
    vc = VitaCounter(0.000180)
    vc.tick(0.000001)
    assert vc.value == pytest.approx(0.000181)
