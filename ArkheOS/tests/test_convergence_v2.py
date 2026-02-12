
import pytest
from arkhe.algebra import vec3
from arkhe.qec import DarvoRepetitionCode
from arkhe.memory import GeodesicMemory
from arkhe.registry import Entity, EntityType, EntityState
from datetime import datetime

def test_vec3_scale_and_project():
    v1 = vec3(10.0, 0.0, 0.0, 0.86, 0.14, 0.00)
    v2 = v1.scale(2.0)
    assert v2.x == 20.0
    assert v2.C == 0.86

    # Project v2 onto v1 (should be v2 itself or along v1)
    v_proj = vec3.project(v2, v1)
    assert v_proj.x == 20.0

def test_darvo_qec():
    code = DarvoRepetitionCode(["N1", "N2", "N3"])

    class MockNode:
        def __init__(self, C, F, omega):
            self.C = C
            self.F = F
            self.omega = omega

    # Case: Node 2 has high fluctuation
    states = {
        "N1": MockNode(0.86, 0.14, 0.00),
        "N2": MockNode(0.50, 0.50, 0.07), # Error
        "N3": MockNode(0.86, 0.14, 0.00)
    }

    syndromes = code.measure_syndrome(states)
    assert syndromes == [0, 1, 0]

    # Correction by majority vote on omega
    node_omegas = {"N1": 0.00, "N2": 0.07, "N3": 0.00}
    target = code.correct(node_omegas)
    assert target == 0.00

def test_geodesic_memory_conflict_resolution():
    memory = GeodesicMemory()

    # Mock entity
    ent = Entity(
        id="1",
        name="Net Profit",
        entity_type=EntityType.FINANCIAL,
        value=1200000.0,
        confidence=0.98,
        last_seen=datetime.utcnow()
    )
    ent.state = EntityState.CONFIRMED

    # Store it
    memory.store_entity(ent)

    # Resolve conflict with same name and similar value
    resolved, value = memory.resolve_conflict("Net Profit", 1200000.0)
    assert resolved == True
    assert value == 1200000.0
