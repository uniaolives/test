
import sys
import os
import numpy as np
import asyncio

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

def test_imports():
    print("Testing foundational imports...")
    from avalon.core.schmidt_bridge import SchmidtBridgeHexagonal
    from avalon.core.verbal_chemistry import VerbalChemistryOptimizer, VerbalStatement
    from avalon.core.hexagonal_water import HexagonalWaterMemory, WaterState
    from avalon.analysis.facial_biofeedback_system import QuantumFacialAnalyzer, QuantumFacialBiofeedback
    from avalon.analysis.verbal_events_processor import VerbalBioCascade
    print("Foundational imports OK.")

    print("Testing requested modules imports...")
    from avalon.analysis.knn_emotion_enhancer import KNNEnhancedFacialBiofeedback
    from avalon.analysis.neural_emotion_engine import UserNeuralProfile
    from avalon.analysis.arkhe_isomorphic_bridge import ArkheIsomorphicLab
    from avalon.analysis.double_exceptionality_detector import DoubleExceptionalityDetector
    print("Requested modules imports OK.")

async def test_instantiation():
    print("Testing instantiations...")
    from avalon.analysis.arkhe_isomorphic_bridge import ArkheIsomorphicLab
    lab = ArkheIsomorphicLab(user_id="test_user")

    from avalon.analysis.double_exceptionality_detector import DoubleExceptionalityDetector
    detector = DoubleExceptionalityDetector()

    # Test clinical prophecy markers
    texts = [
        # Normal high-level text
        "A topologia quântica do ser manifesta-se através de estados de Schmidt altamente correlacionados.",
        # Abstracted Agency / Recursive Rationalization
        "One might assume the body traveled toward the liminal architecture, which aligns with previous interests, explaining the consistent gravity of the situation.",
        # Amnesia / Rupture
        "não lembro de ter postado nada disso ontem. It appears that a different cognitive specialization was active."
    ]
    claims = ["Não lembro de ter postado nada."]

    profile = detector.analyze_2e_profile(texts, claims)
    print(f"2e Profile Detected: {profile['is_double_exceptional']}")
    print(f"DID Indicators: {profile['did_indicators_count']}")
    print(f"VCI-PSI Jaggedness: {profile['vci_psi_gap']:.3f}")
    print(f"Amnesia status: {profile['amnesia_status']}")

    print("Instantiations OK.")

if __name__ == "__main__":
    test_imports()
    asyncio.run(test_instantiation())
