
import sys
import os

# Add ArkheOS/src to path
sys.path.append(os.path.join(os.getcwd(), 'ArkheOS/src'))

from arkhe.preservation import SmartFix
from arkhe.viz import AUV
from arkhe.geodesic import Practitioner

def test_arch_gamma_9052_state():
    # Verify state Γ₉₀₅₂ logic
    vila = AUV.load_snapshot("vila_madalena_20260213")
    assert vila is not None

    fix = SmartFix()
    fix.detect_missing_drive()
    fix.generate_report()

    practitioner = Practitioner.identify()
    assert practitioner.name == "Rafael Henrique"

def test_dual_phi_gamma_9052():
    # Verify 0.9969 convergence
    sys.path.append(os.path.join(os.getcwd(), 'ArkheOS/scripts'))
    from dual_phi_calculator import calculate_phi_system

    # State Γ₉₀₅₂ values
    pk = 1.000
    pf = 0.985
    pg = 1.000
    pb = 0.5625
    pm = 0.5625

    phi_s = calculate_phi_system(pk, pf, pg, pb, pm)
    assert round(phi_s, 4) == 0.9969

if __name__ == "__main__":
    test_arch_gamma_9052_state()
    test_dual_phi_gamma_9052()
    print("Verification of Geodesic Arch (Γ₉₀₅₂): OK")
