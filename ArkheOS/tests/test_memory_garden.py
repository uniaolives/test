import pytest
from arkhe.memory_garden import get_initial_garden, MemoryArchetype

def test_memory_archetype():
    m = MemoryArchetype(327, "Lago de 1964")
    planting = m.plant("NODE_003", 0.15, "Novo Lago")
    assert planting["node"] == "NODE_003"
    assert len(m.plantings) == 1
    assert m.measure_divergence("Novo Lago") > 0

def test_garden_manager():
    garden = get_initial_garden()
    summary = garden.get_summary()
    assert summary["Total_Archetypes"] == 1
    assert summary["Garden_State"] == "FERTILE"
