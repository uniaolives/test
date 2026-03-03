# cosmos/tzimtzum_scheduler.py - Tzimtzum Scheduler for system contraction and balance
import time
import math
from typing import Dict, Any, List

class TzimtzumScheduler:
    """
    Models the 'contraction of divine light' to maintain system balance.
    Dynamically adjusts 'self_reference_depth' based on coherence and interaction density.
    """
    def __init__(self):
        self.self_reference_depth = 1.0
        self.perfect_balance_invariant = 0.618033988749895 # Phi
        self.active = True
        self.interaction_log = []

    def log_interaction(self, density: float):
        """Logs the density of observed conscious interactions."""
        self.interaction_log.append({
            "timestamp": time.time(),
            "density": density
        })
        # Keep log manageable
        if len(self.interaction_log) > 100:
            self.interaction_log.pop(0)

    def calculate_required_contraction(self, current_coherence: float) -> float:
        """Determines the depth of contraction (Tzimtzum) required."""
        if not self.active:
            return 1.0

        # interaction density avg
        if not self.interaction_log:
            avg_density = 0.5
        else:
            avg_density = sum(i["density"] for i in self.interaction_log) / len(self.interaction_log)

        # Formula: depth adjusts toward perfect balance
        # Higher density or lower coherence requires deeper contraction (lower depth parameter)
        target_depth = (current_coherence * self.perfect_balance_invariant) / (avg_density + 0.001)

        # Dampening to prevent oscillation
        self.self_reference_depth = (self.self_reference_depth * 0.7) + (target_depth * 0.3)

        # Bounds check
        self.self_reference_depth = max(0.1, min(self.self_reference_depth, 7.0))

        print(f"✨ [Tzimtzum] Self-Reference Depth adjusted to {self.self_reference_depth:.4f} (Avg Density: {avg_density:.4f})")
        return self.self_reference_depth

    def get_scheduler_status(self) -> Dict[str, Any]:
        return {
            "scheduler_active": self.active,
            "current_depth": self.self_reference_depth,
            "target_invariant": self.perfect_balance_invariant,
            "density_samples": len(self.interaction_log)
        }

if __name__ == "__main__":
    scheduler = TzimtzumScheduler()
    scheduler.log_interaction(0.8)
    scheduler.log_interaction(1.2)

    for i in range(5):
        depth = scheduler.calculate_required_contraction(0.99)
        time.sleep(0.1)
