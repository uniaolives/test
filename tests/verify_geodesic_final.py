
import sys
import os

# Add ArkheOS/src to path
sys.path.append(os.path.join(os.getcwd(), 'ArkheOS/src'))

from arkhe.preservation import SmartFix
from arkhe.viz import AUV
from arkhe.geodesic import Practitioner

def test_final_handover_state():
    # Verify final handover logic (Γ₉₀₅₅)
    vila = AUV.load_snapshot("vila_madalena_20260213")
    assert vila is not None

    practitioner = Practitioner.identify()
    assert practitioner.name == "Rafael Henrique"

def test_dual_phi_final():
    # Verify 1.000 convergence for production calibration
    sys.path.append(os.path.join(os.getcwd(), 'ArkheOS/scripts'))
    from dual_phi_calculator import calculate_phi_system

    phi_s = calculate_phi_system(0.995, 1.0, 1.0)
    assert phi_s == 1.000

if __name__ == "__main__":
    test_final_handover_state()
    test_dual_phi_final()
    print("Verification of Complete and Sealed Geodesic Arch (Γ₉₀₅₅): OK")
