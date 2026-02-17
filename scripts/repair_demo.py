# scripts/repair_demo.py
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from papercoder_kernel.dynamics.inflation import ScaleAwareInflation
from papercoder_kernel.dynamics.repair import MRN_RepairComplex

def main():
    print("--- MRN Repair Complex Demo (Phaistos Disc Suture) ---")

    n_scales = 15
    n_members = 20

    # Generate coherent ensemble
    ensemble = np.random.normal(0, 0.01, (n_members, n_scales))

    # Simulate a "Double Strand Break" at scale 7
    # (Increase variance drastically at this scale)
    ensemble[:, 7] = np.random.normal(0, 5.0, n_members)

    print(f"Ensemble with simulated break at scale 7.")

    # Setup modules
    sai = ScaleAwareInflation(n_scales)
    repair = MRN_RepairComplex(ensemble, sai)

    # 1. Detect Breaks
    breaks = repair.detect_breaks(coherence_threshold=0.3)
    print(f"Detected breaks at scales: {breaks}")

    # 2. Recruit Repair
    if len(breaks) > 0:
        print(f"Recruiting repair for detected breaks...")
        repair.recruit_repair(breaks)
        print(f"Repair log: {repair.repair_log}")

    # 3. Verify Suture
    # Assume we know the ground truth for some fragments
    known_fragments = {7: 0.0, 0: 0.0}
    success = repair.verify_suture(known_fragments)

    print(f"\nSuture verification: {'PASSED' if success else 'FAILED'}")
    print(f"Final mean at scale 7: {np.mean(ensemble[:, 7]):.6f}")

if __name__ == "__main__":
    main()
