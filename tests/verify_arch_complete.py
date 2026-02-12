
import sys
import os

# Add ArkheOS/src to path
sys.path.append(os.path.join(os.getcwd(), 'ArkheOS/src'))

from arkhe.preservation import SmartFix
from arkhe.viz import AUV
from arkhe.geodesic import Practitioner

def test_arch_complete_state():
    # Verify final state Γ₉₀₄₈ logic
    vila = AUV.load_snapshot("vila_madalena_20260213")
    assert vila is not None

    fix = SmartFix()
    fix.detect_missing_drive()
    fix.generate_report()

    practitioner = Practitioner.identify()
    assert practitioner.name == "Rafael Henrique"

def test_dual_phi_final():
    # Verify 1.000 convergence
    sys.path.append(os.path.join(os.getcwd(), 'ArkheOS/scripts'))
    from dual_phi_calculator import calculate_phi_system

    phi_s = calculate_phi_system(1.0, 1.0, 1.0)
    assert phi_s == 1.000

if __name__ == "__main__":
    test_arch_complete_state()
    test_dual_phi_final()
    print("Verification of Complete Geodesic Arch (Γ₉₀₄₈): OK")
