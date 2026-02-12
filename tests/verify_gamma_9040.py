
import sys
import os

# Add ArkheOS/src to path
sys.path.append(os.path.join(os.getcwd(), 'ArkheOS/src'))

from arkhe.preservation import SmartFix
from arkhe.viz import AUV
from arkhe.geodesic import Practitioner

def test_state_gamma_9040():
    # Verify ignition.py logic
    vila = AUV.load_snapshot("vila_madalena_20260213")
    assert vila is not None

    fix = SmartFix()
    fix.detect_missing_drive()
    fix.generate_report()
    assert fix.coherence == 0.9998

    practitioner = Practitioner.identify()
    assert practitioner.name == "Rafael Henrique"
    assert practitioner.hesitation == 47.000

def test_dual_phi_calculator():
    # Import the calculator script logic
    sys.path.append(os.path.join(os.getcwd(), 'ArkheOS/scripts'))
    from dual_phi_calculator import calculate_phi_system

    pk = 1.000
    pf = 0.950
    pg = 0.314

    phi_s = calculate_phi_system(pk, pf, pg)
    assert phi_s == 0.503

if __name__ == "__main__":
    test_state_gamma_9040()
    test_dual_phi_calculator()
    print("Verification of State Γ₉₀₄₀ and Tools: OK")
