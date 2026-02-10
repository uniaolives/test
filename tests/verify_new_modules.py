
import sys
import os
import numpy as np
import asyncio
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

def test_imports():
    print("Testing foundational and cosmic imports...")
    from avalon.core.schmidt_bridge import SchmidtBridgeHexagonal
    from avalon.core.verbal_chemistry import VerbalChemistryOptimizer, VerbalStatement
    from avalon.core.hexagonal_water import HexagonalWaterMemory, WaterState
    from avalon.analysis.facial_biofeedback_system import QuantumFacialAnalyzer, QuantumFacialBiofeedback
    from avalon.analysis.verbal_events_processor import VerbalBioCascade
    from avalon.core.celestial_helix import CosmicDNAHelix
    from avalon.core.celestial_entanglement import CelestialEntanglement
    from avalon.analysis.cosmological_synthesis import HecatonicosachoronConscious, FourDObserver
    from avalon.analysis.celestial_psychometrics import CelestialPsychometrics
    from avalon.analysis.dimensional_consciousness import NeuroCelestialResonance
    from avalon.analysis.neural_emotion_engine import NeuralQuantumAnalyzer
    from avalon.quantum.embeddings import QuantumEmbeddingIntegrator
    from avalon.analysis.arkhe_unified_theory import ArkheConsciousnessBridge
    from avalon.analysis.arkhe_theory import ArkheConsciousnessArchitecture
    print("All imports OK.")

async def test_logic_integration():
    print("Testing logic integration...")

    from avalon.analysis.arkhe_theory import ArkheConsciousnessArchitecture
    arch = ArkheConsciousnessArchitecture()
    profile = arch.initialize_2e_system(0.8, 0.6)
    print(f"Architecture Profile: {profile['system_type']}, Coherence: {profile['arkhe_coherence']:.3f}")

    from avalon.analysis.double_exceptionality_detector import DoubleExceptionalityDetector
    detector = DoubleExceptionalityDetector()

    texts = [
        "The epistemological foundations of quantum consciousness theory necessitate a radical reconceptualization. This aligns with gravity and likely explains the consistent results.",
        "One might assume the body traveled toward the liminal architecture, which aligns with previous interests.",
        "I feel like I am floating in an unreal dream, detached from the bulk."
    ]
    claims = ["Não lembro de ter postado nada.", "I feel scared."]
    celestial_context = {'moon_house': 8, 'psi_coefficient': 0.85}

    profile = detector.analyze_2e_profile(texts, claims, celestial_context=celestial_context)

    print(f"2e Profile Detected: {profile['is_double_exceptional']}")
    print(f"Unified Type: {profile['unified_consciousness']['consciousness_type']}")
    print(f"Bilocation Status: {profile['bilocation_protocol_status']}")
    print(f"Ego Cursor: {profile['ego_cursor_position']}")
    print(f"Detected Masks: {[m['type'] for m in profile['detected_masks']]}")

    # Test Neural Engine Instantiation and Training
    from avalon.analysis.neural_emotion_engine import NeuralQuantumAnalyzer
    analyzer = NeuralQuantumAnalyzer(user_id="cosmic_architect")

    # Simulate some frame analyses to trigger training
    for _ in range(21):
        analysis = {'face_detected': True, 'emotion': 'surprise', 'valence': 0.8, 'arousal': 0.9}
        processed = await analyzer.process_emotional_state_with_neural(analysis)

    print(f"Neural processing active: {analyzer.get_personalized_insights()['model_status']}")
    print(f"Sequences collected: {len(analyzer.user_profile.sequences)}")

    # Test Celestial DNA
    from avalon.core.celestial_helix import CosmicDNAHelix
    dna = CosmicDNAHelix()
    schmidt = dna.to_schmidt_state()
    print(f"Celestial DNA Schmidt Factor: {schmidt.coherence_factor:.3f}")

    # Test Celestial Entanglement
    from avalon.core.celestial_entanglement import CelestialEntanglement
    ent = CelestialEntanglement(dna)
    matrix = ent.calculate_entanglement_matrix()
    print(f"Entanglement Matrix Shape: {matrix.shape}")
    coherence = ent.calculate_quantum_coherence()
    print(f"Celestial Coherence: {coherence['quantum_coherence']:.3f}")

    # Test Isomorphic Bridge
    from avalon.analysis.arkhe_isomorphic_bridge import ArkheIsomorphicLab
    lab = ArkheIsomorphicLab()
    molecule = lab.engine.design_consciousness_molecule(
        target_state="focused_flow",
        user_verbal_input="I am at peak focus."
    )
    print(f"Molecule designed: {molecule.drug_name}")

    print("Logic integration tests OK.")

if __name__ == "__main__":
    test_imports()
    asyncio.run(test_logic_integration())
