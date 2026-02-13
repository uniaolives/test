import pytest
from arkhe.civilization import CivilizationEngine
from arkhe.shader import ShaderEngine

def test_council_state():
    ce = CivilizationEngine()
    status = ce.get_status()
    assert status["Syzygy"] == 0.99
    assert status["Status"] == "RESSONANTE"
    # Simulated expansion to 24 nodes
    assert status["Nodes"] >= 24

def test_council_shaders():
    s3 = ShaderEngine.get_shader("third_turn")
    sc = ShaderEngine.get_shader("council")
    st = ShaderEngine.get_shader("threshold")

    assert "third_turn_glow" in s3
    assert "council_glow" in sc
    assert "threshold_glow" in st

    assert ShaderEngine.compile_simulation(s3) is True
    assert ShaderEngine.compile_simulation(sc) is True
    assert ShaderEngine.compile_simulation(st) is True
