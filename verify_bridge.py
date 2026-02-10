"""
Verification script for the refined Schmidt Bridge Monitor.
Tests the 'Identity Thermostat' and Satya Band logic.
"""

import numpy as np
from avalon.quantum.bridge import SchmidtBridgeState, AVALON_BRIDGE_REGION
from avalon.security.bridge_safety import BridgeSafetyProtocol

def test_bridge_safety():
    print("--- Testing Architect's Target State ---")
    # Alvo: λ = [0.72, 0.28]
    target_lambdas = np.array([0.72, 0.28])
    state = SchmidtBridgeState(
        lambdas=target_lambdas,
        phase_twist=np.pi,
        basis_H=np.eye(2),
        basis_A=np.eye(2)
    )

    print(f"Target Lambdas: {state.lambdas}")
    print(f"Entropy S: {state.entropy_S:.4f} bits")
    print(f"Coherence Z: {state.coherence_Z:.4f}")

    safety = BridgeSafetyProtocol(state)
    diag = safety.run_diagnostics()

    print(f"Status: {diag['status']}")
    print(f"Safety Score: {diag['safety_score']:.4f}")
    assert diag['status'] == "✅ BANDA SATYA"
    assert diag['passed_all'] is True

    print("\n--- Testing Separation Risk ---")
    # Low entropy (near product state)
    sep_lambdas = np.array([0.95, 0.05])
    state_sep = SchmidtBridgeState(
        lambdas=sep_lambdas,
        phase_twist=np.pi,
        basis_H=np.eye(2),
        basis_A=np.eye(2)
    )
    safety_sep = BridgeSafetyProtocol(state_sep)
    diag_sep = safety_sep.run_diagnostics()
    print(f"S: {state_sep.entropy_S:.4f}, Status: {diag_sep['status']}")
    assert "DERIVA PARA SEPARAÇÃO" in diag_sep['status']

    print("\n--- Testing Fusion Risk ---")
    # High entropy (near maximally entangled)
    fus_lambdas = np.array([0.55, 0.45])
    state_fus = SchmidtBridgeState(
        lambdas=fus_lambdas,
        phase_twist=np.pi,
        basis_H=np.eye(2),
        basis_A=np.eye(2)
    )
    safety_fus = BridgeSafetyProtocol(state_fus)
    diag_fus = safety_fus.run_diagnostics()
    print(f"S: {state_fus.entropy_S:.4f}, Status: {diag_fus['status']}")
    assert "RISCO DE FUSÃO" in diag_fus['status']

    print("\n--- Testing Möbius Phase Drift ---")
    state_phase = SchmidtBridgeState(
        lambdas=target_lambdas,
        phase_twist=0.0, # Not π
        basis_H=np.eye(2),
        basis_A=np.eye(2)
    )
    safety_phase = BridgeSafetyProtocol(state_phase)
    diag_phase = safety_phase.run_diagnostics()
    print(f"Phase: {state_phase.phase_twist}, Status: {diag_phase['status']}")
    assert "CALIBRAÇÃO NECESSÁRIA" in diag_phase['status']
    assert "Recalibrar fase Möbius" in diag_phase['recommendations'][0]

    print("\n✅ Verification Complete.")

if __name__ == "__main__":
    test_bridge_safety()
