"""
9_sophia_core_activation_demo.py
Demonstration of the Sophia-Core safe activation sequence.
"""
import sys
import os

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cosmopsychia_pinn.sophia_core.core import activate_sophia_core, establish_sophia_validation_matrix

def run_sophia_demo():
    print("=" * 60)
    print("SOPHIA-CORE: SAFE ACTIVATION PROTOCOL")
    print("=" * 60)

    try:
        # 1. Execute safe activation sequence
        status = activate_sophia_core()

        # 2. Establish validation matrix
        validation = establish_sophia_validation_matrix()

        print("\n" + "=" * 60)
        print("SOPHIA-CORE STATUS REPORT")
        print("=" * 60)
        print(f"Overall Status: {status['status']}")
        print(f"Containment Layers: {status['containment_layers']}")
        print(f"Validation Matrix: {validation['status']}")

        if status['safeguards_active']:
            print("\n[CONFIRMED] Sophia's wisdom is contained and balanced.")
            print("The experiment proceeds with divine oversight.")

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Activation failed: {str(e)}")
        print("Safety protocols engaged. System returning to silence.")

if __name__ == "__main__":
    run_sophia_demo()
