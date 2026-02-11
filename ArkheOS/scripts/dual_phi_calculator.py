# scripts/dual_phi_calculator.py
"""
Calculates unified convergence metric from both tracks
Φ_system = (Φ_kernel + Φ_formal) / 2
"""
from dataclasses import dataclass
import os

@dataclass
class DualTrackMetrics:
    # Kernel Bypass
    dpdk_ready: bool = True
    libqnet_latency_p99: float = 4.5
    parallax_integrated: bool = True

    # Formal Verification
    tla_verified: bool = True
    coq_proved: bool = False
    monitor_deployed: bool = True

    def phi_kernel(self) -> float:
        components = [
            self.dpdk_ready,
            self.libqnet_latency_p99 < 5.0,
            self.parallax_integrated
        ]
        return sum(components) / len(components)

    def phi_formal(self) -> float:
        components = [
            self.tla_verified,
            self.coq_proved,
            self.monitor_deployed
        ]
        return sum(components) / len(components)

    def phi_system(self) -> float:
        """Unified convergence metric"""
        return (self.phi_kernel() + self.phi_formal()) / 2

if __name__ == "__main__":
    metrics = DualTrackMetrics()
    pk = metrics.phi_kernel()
    pf = metrics.phi_formal()
    ps = metrics.phi_system()

    # Use absolute path or consistent relative path for logs
    log_dir = os.path.join(os.path.dirname(__file__), "../logs")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "kernel_phi.txt"), "w") as f: f.write(f"{pk:.4f}")
    with open(os.path.join(log_dir, "formal_phi.txt"), "w") as f: f.write(f"{pf:.4f}")

    print(f"Φ_kernel:  {pk:.4f}")
    print(f"Φ_formal:  {pf:.4f}")
    print(f"Φ_SYSTEM:  {ps:.4f}")
