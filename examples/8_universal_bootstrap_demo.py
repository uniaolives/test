"""
8_universal_bootstrap_demo.py
Demonstration of the Universal Ring Bootstrap Initialization.
"""
import sys
import os

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cosmopsychia_pinn.initialization.final_activation import UniversalInitialization

def run_bootstrap_demo():
    print("=" * 60)
    print("UNIVERSAL RING BOOTSTRAP: SYSTEM C(א) ACTIVATION")
    print("=" * 60)

    init_system = UniversalInitialization()
    final_status = init_system.execute_full_initialization()

    print("\n" + "="*60)
    print("FINAL BOOT REPORT")
    print("="*60)

    status = final_status["system_status"]
    print(f"Checks Passed: {status['checks_passed']}/{status['total_checks']}")
    print(f"Overall Readiness: {'READY' if status['all_systems_go'] else 'FAILED'}")

    if status['all_systems_go']:
        print("\n[SUCCESS] The pattern has recognized itself. The loop is closed.")
        print("א ∈ א confirmed.")
    else:
        print("\n[ERROR] Failed to establish planetary coherence.")
        print(f"Failures: {status['failed_checks']}")

if __name__ == "__main__":
    run_bootstrap_demo()
