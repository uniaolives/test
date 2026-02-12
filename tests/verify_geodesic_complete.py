
import sys
import os

# Add ArkheOS/src to path
sys.path.append(os.path.join(os.getcwd(), 'ArkheOS/src'))

from arkhe.preservation import SmartFix
from arkhe.viz import AUV
from arkhe.geodesic import Practitioner

def test_final_state_gamma_9054():
    # Verify final handover logic
    vila = AUV.load_snapshot("vila_madalena_20260213")
    assert vila is not None

    practitioner = Practitioner.identify()
    assert practitioner.name == "Rafael Henrique"

def test_dual_phi_absolute():
    # Verify 1.000 convergence
    sys.path.append(os.path.join(os.getcwd(), 'ArkheOS/scripts'))
    from dual_phi_calculator import calculate_phi_system

    phi_s = calculate_phi_system(1.0, 1.0, 1.0, 1.0, 1.0)
    assert phi_s == 1.000

if __name__ == "__main__":
    test_final_state_gamma_9054()
    test_dual_phi_absolute()
    print("Verification of Complete Geodesic Arch (Γ₉₀₅₄ - 100%): OK")
