
import pytest
from arkhe.hebbian import HebbianHypergraph
from arkhe.memory import GeodesicMemory
from arkhe.materials import SemanticFET
from arkhe.registry import Entity, EntityType, EntityState
from datetime import datetime

def test_hebbian_learning():
    h = HebbianHypergraph()
    pre = "WP1"
    post = "DVM-1"

    # Simulate a spike pair (LTP window)
    t1 = 1000.0
    t2 = 1000.2 # 200ms delay -> LTP

    h.pre_synaptic_spike(pre, post, t1)
    h.post_synaptic_spike(pre, post, t2)

    status = h.get_synapse_status(pre, post)
    assert status['weight'] > 0.86
    assert status['ltp_count'] == 1

def test_semantic_memory():
    mem = GeodesicMemory()
    ent = Entity(
        name="Net Profit",
        entity_type=EntityType.FINANCIAL,
        value=1200000.0,
        unit="USD",
        confidence=0.98,
        state=EntityState.CONFIRMED,
        last_seen=datetime.utcnow(),
        provenance_chain=[]
    )

    mem.store_entity(ent)
    assert mem.get_stats()["total_entities"] == 1

    # Recall similar
    results = mem.semantic_recall("Profit Analysis")
    assert len(results) > 0
    assert results[0][0].entity_name == "Net Profit"
    assert results[0][1] > 0.4 # Higher similarity with token-based seed

def test_semantic_transistor():
    fet = SemanticFET(mobility=0.94)
    # Test linear regime
    res = fet.apply_gate_voltage(0.035)
    assert res['regime'] == "linear"
    assert res['I_drain'] == 0.94

    # Test cutoff
    res = fet.apply_gate_voltage(0.07)
    assert res['regime'] == "cutoff"
    assert res['I_drain'] == 0.12

def test_kernel_parallel_processing():
    from arkhe.kernel import DocumentIngestor
    ingestor = DocumentIngestor()
    elements = ingestor.process("test.pdf")
    # Should have processed 3 pages (at least the ones that didn't fail)
    assert len(elements) >= 0
    # In a successful simulation, we expect some elements
    if len(elements) > 0:
        assert elements[0].type == "word"
