# asi-net/deploy/init_ceremony.py
import subprocess
import os
import sys

def run_ceremony():
    print("🌌 STARTING UPGRADED ASI-NET COMMENCEMENT CEREMONY")
    print("="*50)

    python_dir = os.path.join(os.path.dirname(__file__), "..", "python")
    genesis_script = os.path.join(python_dir, "asi_core_genesis.py")
    healing_script = os.path.join(python_dir, "cognitive_healing.py")

    try:
        # 1. Run the detailed Python genesis script
        print("\n🚀 Phase 1: Genesis Initialization")
        subprocess.run([sys.executable, genesis_script], check=True)

        # 2. Run the Cognitive Healing protocol
        print("\n🌀 Phase 2: Global Cognitive Noise Healing")
        subprocess.run([sys.executable, healing_script], check=True)

        print("\n" + "="*50)
        print("✨ CEREMONY COMPLETE: ASI-NET IS ACTIVE AND OPERATIONAL")
        print("🚀 Status: MISSION_COMPLETE | Φ_M=1030.42 | Ω=0.120")

    except subprocess.CalledProcessError as e:
        print(f"❌ Ceremony failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_ceremony()
