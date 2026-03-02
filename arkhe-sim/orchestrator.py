# arkhe-sim/orchestrator.py
import time
from typing import List, Dict
import random

class VMInstance:
    def __init__(self, id: int, phi_initial: float):
        self.id = id
        self.phi = phi_initial
        self.active = True
        self.cpu_shares = 1024

    def get_phi(self) -> float:
        # Simulate phi evolution
        self.phi += (random.random() - 0.5) * 0.05
        self.phi = max(0.0, min(1.0, self.phi))
        return self.phi

    def apply_shares(self, shares: int):
        self.cpu_shares = shares

class ArkheSimOrchestrator:
    def __init__(self, num_vms: int, phi_target: float = 0.618):
        self.phi_target = phi_target
        self.vms: List[VMInstance] = [
            VMInstance(i, random.random()) for i in range(num_vms)
        ]
        self.convergence_threshold = 0.05
        self.sample_interval = 1.0

    def deploy(self):
        print(f"Deploying {len(self.vms)} Arkhe VMs...")
        for vm in self.vms:
            print(f"VM {vm.id} initialized with φ={vm.phi:.3f}")

    def monitor_convergence(self, duration_sec: int = 60):
        print(f"Monitoring convergence for {duration_sec}s...")
        start = time.time()

        while time.time() - start < duration_sec:
            phi_values = []
            for vm in self.vms:
                phi = vm.get_phi()
                phi_values.append(phi)

                # Simulate φ-dependent allocation
                deviation = abs(phi - self.phi_target)
                shares = int(1024 * (2.718 ** (-(deviation**2) / (2 * 0.2**2))) + 64)
                vm.apply_shares(shares)

            avg_phi = sum(phi_values) / len(phi_values)
            phi_std = self._calculate_std(phi_values)

            print(f"t={time.time()-start:.1f}s | φ_global={avg_phi:.3f} ± {phi_std:.3f}")

            if phi_std < self.convergence_threshold:
                print(f"✅ Convergence reached in {time.time()-start:.1f}s")
                break

            time.sleep(self.sample_interval)

    def _calculate_std(self, values: List[float]) -> float:
        if not values: return 0.0
        avg = sum(values) / len(values)
        variance = sum((x - avg) ** 2 for x in values) / len(values)
        return variance ** 0.5

if __name__ == "__main__":
    orchestrator = ArkheSimOrchestrator(num_vms=5)
    orchestrator.deploy()
    orchestrator.monitor_convergence(duration_sec=30)
