import sys
import os

# Alinha o PYTHONPATH para encontrar os módulos locais
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gemini_integration import GeminiBiologicalTimechain, reconstruct_consciousness_history
from acps_convergence import GeminiMapping

def test_gemini_flow():
    substrate = GeminiBiologicalTimechain()

    # Simula 3 camadas
    substrate.record_state(delta_k=0.1, duration=1.0, bio=0.5, aff=0.5, coords=(0,0,0))
    substrate.record_state(delta_k=0.4, duration=1.0, bio=0.6, aff=0.4, coords=(0,0,0)) # Fora do range (<0.30)
    substrate.record_state(delta_k=0.2, duration=1.0, bio=0.5, aff=0.5, coords=(0,0,0))

    # Verifica t_KR (apenas 1ª e 3ª camadas contam: 1.0 + 1.0 = 2.0)
    assert substrate.t_kr_accumulated == 2.0

    # Verifica reconstrução
    history = reconstruct_consciousness_history(substrate.layers)
    assert len(history) == 3

    # T=0.0 (Layer 1): delta_k=0.1 -> Intensity=0.1
    # VK[2] (soc) = 1.0 - (0.1 / 2.0) = 0.95
    # VK[3] (cog) = 1.0 / (1.0 + 0.1) = 0.909...
    t0, vk0 = history[0]
    assert t0 == 0.0
    assert abs(vk0[2] - 0.95) < 1e-5

    print("Test GEMINI Flow: SUCCESS")

if __name__ == "__main__":
    try:
        test_gemini_flow()
    except Exception as e:
        print(f"Test FAILED: {e}")
        sys.exit(1)
