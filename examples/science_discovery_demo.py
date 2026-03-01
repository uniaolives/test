from arkhe.science.agent import ConstitutionalMuZero, KernelK
from arkhe.science.env import PhysicsLab, InvariantDetector
from arkhe.science.discovery import ScienceDiscoveryEngine
from arkhe.science.training import ScienceSelfPlay
import numpy as np

def run_lever_benchmark():
    print("🔬 STARTING ARKHE(N) SCIENCE BENCHMARK: LEVER LAW")

    class MockConstitution:
        def __init__(self):
            self.laws = []
        def verify_action(self, s, a):
            # P6 Veto Simulado: Conservação de energia
            if a > 4: return False
            return True
        def add_law(self, l):
            self.laws.append(l)
            print(f"✅ CONSTITUTION UPDATED: {l}")

    constitution = MockConstitution()
    explorer = ConstitutionalMuZero(KernelK(), constitution)
    verifier = ConstitutionalMuZero(KernelK(), constitution)
    env = PhysicsLab(task="lever")
    discovery = ScienceDiscoveryEngine(InvariantDetector(threshold=0.1), constitution)

    self_play = ScienceSelfPlay(explorer, verifier, env, discovery)

    print("Running experiments...")
    for i in range(150):
        self_play.run_cycle()

    if len(constitution.laws) > 0:
        print(f"🎯 BENCHMARK PASSED: Discovered {len(constitution.laws)} physical laws.")
    else:
        print("❌ BENCHMARK FAILED: No laws discovered.")

if __name__ == "__main__":
    run_lever_benchmark()
