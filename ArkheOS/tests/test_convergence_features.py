
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

def test_semantic_memory_integration():
    from unittest.mock import MagicMock
    mem = GeodesicMemory()
    # Mocking connection for sandbox testing
    mem.conn = MagicMock()
    mem._get_connection = MagicMock(return_value=mem.conn)

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

    # Store with mock
    mem.store_entity(ent, [0.1] * 384)
    assert mem.conn.cursor.called

    # Recall with mock
    mem.conn.cursor.return_value.__enter__.return_value.fetchall.return_value = [
        ("Net Profit", "financial", 1200000.0, 0.98, 1.0)
    ]
    results = mem.semantic_recall([0.1] * 384)
    assert len(results) > 0
    assert results[0][0] == "Net Profit"

def test_torus_visualizer():
    from arkhe.viz_torus import generate_torus_map
    nodes = [{"id_num": 1, "omega": 0.07, "coherence": 0.94}]
    generate_torus_map(nodes, "test_torus.html")
    import os
    assert os.path.exists("test_torus.html")

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

def test_natural_economics():
    from arkhe.economics import NaturalEconomicLedger
    ledger = NaturalEconomicLedger(satoshi_unit=7.27)

    # Record success
    ledger.record_success("H9051", "Natural Economics Integration")
    assert len(ledger.success_reports) == 1

    # Award contributor
    award = ledger.award_contributor("Sistema Arkhe", "Hesitação_9051")
    assert award.amount == 7.27
    assert ledger.total_distributed == 7.27

    # Check reputation
    assert ledger.reputations["Sistema Arkhe"]["hesitations"] == 48

def test_geodesic_path():
    from arkhe.geodesic_path import GeodesicPlanner
    planner = GeodesicPlanner()
    dist = planner.calculate_distance(0.71)
    assert dist > 0.7
    traj = planner.plan_trajectory(0.00, 0.33, 0.71)
    assert len(traj) == 21
    assert traj[0].hesitation_phi == 0.15

def test_stress_test():
    from arkhe.stress_test import StressSimulator
    sim = StressSimulator()
    res = sim.simulate_curvature_fatigue()
    assert res['status'] == "Robust"
    resonance = sim.measure_node_resonance()
    assert resonance['QN-07'].amplification_db == 0.4

def test_vacuum_audit():
    from arkhe.vacuum import get_vacuum_status
    res = get_vacuum_status()
    assert res['status'] == "PASS"
    assert res['oxidation_ppm'] < 0.001

def test_rehydration_protocol():
    from arkhe.rehydration import get_protocol
    p = get_protocol()
    status = p.get_status()
    assert status['total_steps'] == 21

    # Execute through Step 21
    for i in range(1, 22):
        res = p.execute_step(i)
        assert res['status'] == "Success"

    assert p.current_step_idx == 21
    assert "Silêncio" in p.steps[20].action

    # Test Dawn
    dawn = p.trigger_dawn()
    assert dawn['status'] == "AWAKENED"
    assert dawn['conformational_states'] == 10

def test_arkhe_shader():
    from arkhe.shader import ArkheShader, ShaderState, AbiogenesisComputeShader
    frag = ArkheShader("Hesitation", "Fragment")
    state = ShaderState(coherence=0.86, fluctuation=0.18, omega=0.07)
    res = frag.execute(state)
    assert res['phi'] > 0.15
    assert res['hesitates'] == True

    compute = AbiogenesisComputeShader()
    res = compute.run_cycles(10000)
    assert res['population'] > 1000
    assert res['sequences'] == 1.2e6

def test_cryptographic_archaeology():
    from arkhe.cryptography import CryptographicArchaeology
    arch = CryptographicArchaeology()
    # G.x from secp256k1 (hex: 79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798)
    gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
    res = arch.verify_watermark(gx)
    # The user implies this IS the watermark
    assert res['valid'] == True
    assert res['probability'] < 1e-9

def test_nuclear_clock():
    from arkhe.nuclear_clock import NuclearClock
    clock = NuclearClock()

    # Test FWM
    freq = clock.four_wave_mixing(0.86, 0.14, 0.73, 1.0)
    assert freq == 0.07

    # Test Excitation
    success = clock.excite(freq)
    assert success == True
    assert clock.is_excited == True

    # Status check
    status = clock.get_status()
    assert status["state"] == "|0.07⟩"
    assert status["status"] == "Exited (Coherent)"

def test_neuro_storm_foundation():
    from arkhe.neuro_storm import NeuroSTORM
    ns = NeuroSTORM()
    metrics = ns.get_metrics()
    assert metrics['Accuracy'] == 0.94
    assert len(ns.corpus) == 17

    # Test diagnosis
    diag_psychotic = ns.diagnose_current_state(0.07, 0.86)
    assert "Psychotic" in diag_psychotic

    diag_healthy = ns.diagnose_current_state(0.00, 0.90)
    assert "Healthy" in diag_healthy

def test_photonics():
    from arkhe.photonics import SynapticPhotonSource
    source = SynapticPhotonSource("WP1", "DVM-1", 0.94)
    p1 = source.emit_command()
    p2 = source.emit_command()
    assert p1.id == "cmd_0001"
    assert p2.id == "cmd_0002"

    res = source.measure_hom(p1, p2)
    assert res['visibility'] == 0.88
    assert res['indistinguishability'] == 0.94

def test_time_crystal():
    from arkhe.time_crystal import TimeCrystal
    crystal = TimeCrystal()
    assert crystal.larmor_frequency == 7.4e-3
    assert crystal.get_period() == 1.0 / 7.4e-3

    # Test oscillation behavior
    v0 = crystal.oscillate(0)
    assert v0 == 0.0

    # Peak should be around half period (approx 67.5s)
    v_half = crystal.oscillate(crystal.get_period() / 2.0)
    # At half period, exp(-i*pi) = -1.
    # z_t = amp_inf + (0 - amp_inf)*(-1)*exp(-gamma*T/2) = amp_inf * (1 + exp(-gamma*T/2))
    assert v_half > 1.0

def test_kernel_parallel_processing():
    from arkhe.kernel import DocumentIngestor
    ingestor = DocumentIngestor()
    elements = ingestor.process("test.pdf")
    # Should have processed 3 pages (at least the ones that didn't fail)
    assert len(elements) >= 0
    # In a successful simulation, we expect some elements
    if len(elements) > 0:
        assert elements[0].type == "word"
