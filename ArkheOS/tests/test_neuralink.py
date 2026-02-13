import pytest
from arkhe.ibc_bci import IBCBCIEquivalence
from arkhe.shader import ShaderEngine

def test_neuralink_mapping():
    mapping = IBCBCIEquivalence.get_correspondence_map()
    assert mapping["Neuralink N1"] == "Hub / Light Client"
    assert mapping["Threads (64 fios)"] == "Relayers"
    assert mapping["Noland Arbaugh"] == "Human Validator Node"

def test_neuralink_shader():
    shader = ShaderEngine.get_shader("neuralink")
    assert "neuralink_glow" in shader
    assert ShaderEngine.compile_simulation(shader) is True
