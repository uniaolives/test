import pytest
from arkhe.civilization import CivilizationEngine
from arkhe.shader import ShaderEngine

def test_council_state():
    ce = CivilizationEngine()
    status = ce.get_status()
    # Test for presence of key metrics
    assert "Syzygy" in status or "Syzygy_Global" in status
    assert status["PHI"] == 0.951
    assert status["Nodes"] >= 4

def test_council_shaders():
    s3 = ShaderEngine.get_shader("dawn")
    sc = ShaderEngine.get_shader("council")
    st = ShaderEngine.get_shader("threshold")

    assert "horizon_color" in s3 or "// Shader not found" not in s3
