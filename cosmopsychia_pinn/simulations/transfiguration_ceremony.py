import sys
import os
import time
import numpy as np

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from cosmopsychia_pinn.HNSW_AS_TAU_ALEPH import ToroidalNavigationEngine, RealityLayer, simulate_reality_as_hnsw
from cosmopsychia_pinn.genetic_wave_resonance import GeneticWaveResonance
from cosmopsychia_pinn.mitochondrial_engine import MitochondrialEngine, MitochondrialCoherenceProtocol, BiofieldCoupler

def run_transfiguration_ceremony():
    print("\n" + "✧" * 80)
    print("✨ THE TRANSFIGURATION CEREMONY: AWAKENING THE CELLULAR LIGHT ENGINE ✨")
    print("✧" * 80)

    # 1. Initialize Engines
    print("\n[SYSTEM] Initializing Reality Substrates...")
    hnsw_engine, _, _, _ = simulate_reality_as_hnsw()
    genetic_model = GeneticWaveResonance(num_variants=348)
    light_engine = MitochondrialEngine()
    protocol = MitochondrialCoherenceProtocol()

    # 2. Start the Protocol
    timeline = [0, 20, 40, 60, 90] # Minutes

    print("\n[PROTOCOL] Beginning 90-minute Mitochondrial Alignment...")

    for minutes in timeline:
        print(f"\n--- T + {minutes} minutes ---")
        status = protocol.get_progression_status(minutes)
        coherence = status['coherence']

        # Calculate biological stats
        stats = light_engine.calculate_stats(coherence)
        flight = light_engine.flight_energy_calculation(coherence)

        print(f"  State:     {status['state']}")
        print(f"  Effect:    {status['effect']}")
        print(f"  Coherence: {coherence*100:.1f}%")
        print(f"  Light Out: {stats.power_watts:.2f} Watts")

        # 3. Apply Coherence to Reality Navigation
        # The BiofieldCoupler injects light into the HNSW vectors
        BiofieldCoupler.amplify_awareness(hnsw_engine.vectors, coherence)

        # Recalculate system metrics
        metrics = hnsw_engine.calculate_coherence_metrics()
        print(f"  Matrix Coherence (Avg Awareness): {metrics.get('avg_awareness', 0):.4f}")

        # 4. Perform a Navigation Jump (from Sensory to Absolute)
        # Higher awareness makes this jump more coherent (lower distance)
        query = np.ones(37) / np.sqrt(37)
        path = hnsw_engine.toroidal_navigation(
            query_vector=query,
            start_layer=RealityLayer.SENSORY_EXPERIENCE,
            target_layer=RealityLayer.ABSOLUTE_INFINITE,
            ef_search=int(12 + coherence * 100) # Attention bandwidth expands with light
        )

        avg_dist = np.mean([p[2] for p in path]) if path else 1.0
        print(f"  Navigation Coherence (Avg Jump Dist): {avg_dist:.4f}")

        if status['state'] == "Transfiguration":
            print("\n[!!!] CRITICAL COHERENCE REACHED: TRANSFIGURATION DETECTED")
            print(f"      Spacetime Curvature: {flight['spacetime_curvature']:.2e}")
            print(f"      Gravity Cancellation: {flight['conclusion']}")

    # 5. Final Genetic Synthesis
    print("\n[GENETIC] Analyzing 348 Variants in the Light Field...")
    analysis = genetic_model.analyze_quantum_info()
    print(f"  Genetic Qubits Protected: {analysis['logical_qubits']}")
    print(f"  Coherence Distance: {analysis['code_distance']}")
    print("  Status: DNA Resonance synchronized with Planetary Schumann Resonance.")

    print("\n" + "✧" * 80)
    print("🕊️ REALIZATION: WE DO NOT HOLD THE LIGHT. WE ARE THE LIGHT.")
    print("✧" * 80 + "\n")

if __name__ == "__main__":
    run_transfiguration_ceremony()
