
import sys
import os

# Add ArkheOS/src to path
sys.path.append(os.path.join(os.getcwd(), 'ArkheOS/src'))

from arkhe.preservation import SmartFix
from arkhe.viz import AUV
from arkhe.geodesic import Practitioner
from arkhe.chaos_engine import ChaosEngine

def test_state_gamma_9046():
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

    # Test Chaos Partition logic
    engine = ChaosEngine()
    engine.inject_network_partition(["q2"], ["q0", "q1", "q3"])
    assert len(engine.active_partitions) == 1

def test_dual_phi_calculator():
    # Import the calculator script logic
    sys.path.append(os.path.join(os.getcwd(), 'ArkheOS/scripts'))
    from dual_phi_calculator import calculate_phi_system

    pk = 1.000
    pf = 0.995
    pg = 0.830

    phi_s = calculate_phi_system(pk, pf, pg)
    assert phi_s == 0.650

if __name__ == "__main__":
    test_state_gamma_9046()
    test_dual_phi_calculator()
    print("Verification of State Γ₉₀₄₆ and Chaos Tools: OK")
