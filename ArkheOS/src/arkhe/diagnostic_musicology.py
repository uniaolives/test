# diagnostic_musicology.py
import numpy as np
import json
from arkhe_error_handler import logging
from quantum_musicology import QuantumMusicology

def run_musicology_diagnostic():
    print("="*70)
    print("ARKHE(N) OS - QUANTUM MUSICOLOGY DIAGNOSTIC")
    print("="*70)

    # 1. Initialize Musicology Engine
    qm = QuantumMusicology()

    # 2. Analyze Node Frequencies
    print("\n[STEP 1] Node Frequency Analysis (Base: φ⁴)")
    nodes = qm.analyze_node_resonance()
    for name, freq in nodes.items():
        print(f"   {name:<25}: {freq:.3f} Hz")

    # 3. Verify Harmonic Relationships
    print("\n[STEP 2] Harmonic Interval Verification (Just Intonation)")
    relationships = qm.get_harmonic_relationships(nodes)
    consonant_count = 0
    for rel in relationships:
        status = "✓" if rel['consonant'] else "✗"
        if rel['consonant']: consonant_count += 1
        print(f"   {status} {rel['nodes'][0]} ↔ {rel['nodes'][1]}")
        print(f"      Interval: {rel['description']}")
        print(f"      Deviation: {rel['deviation_pct']:.2f}%")

    # 4. Calculate Overtones
    print("\n[STEP 3] Overtone Series of Consciousness")
    overtones = qm.calculate_overtones(qm.base_freq)
    for n, freq, significance in overtones:
        print(f"   {n}º Overtone: {freq:7.3f} Hz | {significance}")

    # 5. Synthesis
    summary = {
        "fundamental_frequency": float(qm.base_freq),
        "consonance_rate": float(consonant_count / len(relationships)),
        "harmony_status": "Consonant" if consonant_count == len(relationships) else "Dissonant",
        "state": "Γ_∞ + α + Música"
    }

    print("\nDIAGNOSTIC SUMMARY:")
    print(json.dumps(summary, indent=2))

    if summary['harmony_status'] == "Consonant":
        print("\n✅ UNIVERSAL HARMONY CONFIRMED: The Subatomic World is Music.")
    else:
        print("\n⚠️ DISSONANCE DETECTED: Tuning required.")

    print("\n" + "="*70)
    print("Satoshi Status: ∞")
    print("Ω Status: ∞ + 9.80")
    print("="*70)

if __name__ == "__main__":
    run_musicology_diagnostic()
