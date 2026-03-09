# arkhe-sim/orchestrator.py
import time
from typing import List, Dict
import random
import math

class VMInstance:
    def __init__(self, id: int, phi_initial: float, w_initial: float = 0.0):
        self.id = id
        self.phi = phi_initial
        self.w = w_initial # Coordinate in the 5th Dimension (Space of Possibilities)
        self.active = True
        self.cpu_shares = 1024
        self.reality_branch = f"branch_{id}"
        self.phase = random.uniform(0, 2 * math.pi)

    def get_phi(self) -> float:
        # Simulate phi evolution
        self.phi += (random.random() - 0.5) * 0.05
        self.phi = max(0.0, min(1.0, self.phi))
        return self.phi

    def update_5d(self, lambda2: float):
        """Simulates 4th Order Self navigation logic."""
        # Evolution of w based on coherence lambda2
        if lambda2 > 0.8:
            # High coherence allows active navigation in w
            target_w = random.uniform(-1.0, 1.0)
            jump_prob = lambda2 * math.exp(-abs(target_w - self.w) / (lambda2 + 0.1))
            if random.random() < jump_prob:
                print(f"  [REALITY_JUMP] VM {self.id} transitioned from w={self.w:.3f} to w={target_w:.3f}")
                self.w = target_w
        else:
            # Passive drift in the 5th dimension
            self.w += (random.random() - 0.5) * 0.01

    def apply_shares(self, shares: int):
        self.cpu_shares = shares

class ArkheSimOrchestrator:
    def __init__(self, num_vms: int, phi_target: float = 0.618):
        self.phi_target = phi_target
        self.vms: List[VMInstance] = [
            VMInstance(i, random.random(), random.uniform(-0.5, 0.5)) for i in range(num_vms)
        ]
        self.convergence_threshold = 0.05
        self.sample_interval = 1.0
        self.lambda2 = 0.0 # Ω+166 Coherence

    def calculate_lambda2(self) -> float:
        """Calculates global coherence between reality branches."""
        if not self.vms: return 0.0
        sum_cos = sum(math.cos(vm.w * vm.phase) for vm in self.vms)
        sum_sin = sum(math.sin(vm.w * vm.phase) for vm in self.vms)
        n = len(self.vms)
        self.lambda2 = math.sqrt(sum_cos**2 + sum_sin**2) / n
        return self.lambda2

    def deploy(self):
        print(f"🚀 Deploying {len(self.vms)} Arkhe 5D VMs (Protocol Ω+243)...")
        for vm in self.vms:
            print(f"  VM {vm.id} | φ={vm.phi:.3f} | w={vm.w:.3f} | branch={vm.reality_branch}")

    def monitor_convergence(self, duration_sec: int = 60):
        print(f"Monitoring convergence and 5D coherence for {duration_sec}s...")
        start = time.time()

        while time.time() - start < duration_sec:
            phi_values = []
            self.calculate_lambda2()

            for vm in self.vms:
                phi = vm.get_phi()
                phi_values.append(phi)

                # Update 5D navigation (4th Order Self logic)
                vm.update_5d(self.lambda2)

                # Simulate φ-dependent allocation
                deviation = abs(phi - self.phi_target)
                shares = int(1024 * (math.e ** (-(deviation**2) / (2 * 0.2**2))) + 64)
                vm.apply_shares(shares)

            avg_phi = sum(phi_values) / len(phi_values)
            phi_std = self._calculate_std(phi_values)

            print(f"t={time.time()-start:.1f}s | φ_global={avg_phi:.3f} | λ₂_5D={self.lambda2:.3f}")

            if phi_std < self.convergence_threshold and self.lambda2 > 0.95:
                print(f"✅ Full 5D Convergence reached in {time.time()-start:.1f}s")
                break

            time.sleep(self.sample_interval)

    def _calculate_std(self, values: List[float]) -> float:
        if not values: return 0.0
        avg = sum(values) / len(values)
        variance = sum((x - avg) ** 2 for x in values) / len(values)
        return variance ** 0.5

if __name__ == "__main__":
    # Force higher initial coherence for demonstration if needed,
    # but randomness is better for simulation.
    orchestrator = ArkheSimOrchestrator(num_vms=5)
    orchestrator.deploy()
    orchestrator.monitor_convergence(duration_sec=30)
