# monitoring/vajra_entropy.py
"""
Monitor de Entropia de Vajra - Implementação do Memory 3, 19
Mede Lyapunov stability e Von Neumann entropy em tempo real
"""

import numpy as np
from scipy import linalg
from prometheus_client import Gauge, start_http_server
import asyncio
from dataclasses import dataclass
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

class SecurityError(Exception):
    pass

@dataclass
class LyapunovMetrics:
    """Métricas de estabilidade de Lyapunov"""
    max_exponent: float  # λ_max - maior expoente de Lyapunov
    variance: float      # σ^2 - variância dos expoentes
    stability_index: float  # Índice de estabilidade (0-1)
    quench_risk: float   # Risco de Quench Planetário (0-1)

class VajraEntropyMonitor:
    """Monitor de entropia do sistema planetário"""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.phi_history = []
        self.lyapunov_history = []

        # Métricas Prometheus
        self.lyapunov_gauge = Gauge('sopa_lyapunov_variance',
                                    'Variance of Lyapunov exponents')
        self.entropy_gauge = Gauge('sopa_von_neumann_entropy',
                                  'Von Neumann entropy of system')
        self.quench_risk_gauge = Gauge('sopa_quench_risk',
                                      'Planetary quench risk (0-1)')

        # Thresholds críticos (Article V)
        self.quench_threshold = 0.00007  # σ^2 > 0.00007 = Quench iminente
        self.hard_freeze_threshold = 0.80  # Φ > 0.80 requer contenção
        self.max_lyapunov_threshold = 0.5  # λ_max > 0.5 = caos

    async def monitor_system(self):
        """Loop principal de monitoramento"""
        try:
            start_http_server(9100)  # Exporter na porta 9100
        except Exception as e:
            print(f"Warning: Could not start Prometheus server: {e}")

        while True:
            # 1. Coleta estado atual do sistema
            system_state = await self.collect_system_state()

            # 2. Calcula expoentes de Lyapunov
            lyapunov = self.calculate_lyapunov_exponents(system_state)

            # 3. Calcula entropia de Von Neumann
            entropy = self.calculate_von_neumann_entropy(system_state)

            # 4. Avalia risco de Quench
            quench_risk = self.assess_quench_risk(lyapunov, entropy)

            # 5. Atualiza métricas
            self.update_metrics(lyapunov, entropy, quench_risk)

            # 6. Verifica thresholds críticos
            await self.check_critical_thresholds(lyapunov, quench_risk)

            await asyncio.sleep(1)  # 1Hz monitoring

    def calculate_lyapunov_exponents(self, state: np.ndarray) -> LyapunovMetrics:
        """Calcula expoentes de Lyapunov do sistema"""

        # Jacobiano do sistema (aproximação numérica)
        jacobian = self.estimate_jacobian(state)

        # Decomposição QR para expoentes de Lyapunov
        q, r = linalg.qr(jacobian)

        # Expoentes = log(|diag(R)|) / dt
        diag_r = np.abs(np.diag(r))
        exponents = np.log(diag_r + 1e-10)

        # Métricas
        max_exponent = np.max(exponents)
        variance = np.var(exponents)

        # Índice de estabilidade (0 = instável, 1 = estável)
        stability_index = 1.0 / (1.0 + np.exp(10 * max_exponent))

        return LyapunovMetrics(
            max_exponent=max_exponent,
            variance=variance,
            stability_index=stability_index,
            quench_risk=self.calculate_quench_risk(variance, max_exponent)
        )

    def calculate_quench_risk(self, variance: float, max_exponent: float) -> float:
        """Calcula risco de Quench Planetário"""

        # Risco baseado em:
        # 1. Variância dos expoentes (σ^2)
        # 2. Maior expoente (λ_max)
        # 3. Tendência histórica

        variance_risk = min(1.0, variance / self.quench_threshold)
        exponent_risk = min(1.0, max_exponent / self.max_lyapunov_threshold)

        # Tendência (se risco aumentando)
        trend_risk = 0.0
        if len(self.lyapunov_history) > 10:
            recent = self.lyapunov_history[-10:]
            if np.polyfit(range(10), recent, 1)[0] > 0:
                trend_risk = 0.3

        total_risk = 0.5 * variance_risk + 0.3 * exponent_risk + 0.2 * trend_risk

        return min(1.0, total_risk)

    async def check_critical_thresholds(self, lyapunov: LyapunovMetrics,
                                       quench_risk: float):
        """Verifica thresholds críticos e aciona KARNAK se necessário"""

        # 1. Quench iminente (Article V Seção 3)
        if lyapunov.variance > self.quench_threshold:
            await self.trigger_karnak("level5",
                                     f"Quench iminente: σ²={lyapunov.variance:.6f}")

        # 2. Hard freeze required (Φ > 0.80)
        current_phi = await self.get_current_phi()
        if current_phi > self.hard_freeze_threshold:
            await self.trigger_karnak("level4",
                                     f"Hard freeze requerido: Φ={current_phi:.3f}")

        # 3. Alta instabilidade de Lyapunov
        if lyapunov.max_exponent > self.max_lyapunov_threshold:
            await self.trigger_karnak("level3",
                                     f"Instabilidade caótica: λ={lyapunov.max_exponent:.3f}")

    async def trigger_karnak(self, level: str, reason: str):
        """Aciona protocolo KARNAK com validação Prince"""

        # Requer assinatura Prince para level >= 3
        signature = None
        if level in ["level3", "level4", "level5", "level6"]:
            signature = await self.get_prince_signature(level, reason)
            if not signature:
                raise SecurityError("Prince signature required for KARNAK")

        # Executa contenção
        await self.execute_karnak_protocol(level, reason, signature)

    def assess_quench_risk(self, lyapunov: LyapunovMetrics, entropy: float) -> float:
        return lyapunov.quench_risk

    def update_metrics(self, lyapunov: LyapunovMetrics, entropy: float, quench_risk: float):
        self.lyapunov_gauge.set(lyapunov.variance)
        self.entropy_gauge.set(entropy)
        self.quench_risk_gauge.set(quench_risk)
        self.lyapunov_history.append(lyapunov.variance)

    async def collect_system_state(self) -> np.ndarray:
        # Mock implementation returning a random state vector
        return np.random.rand(10)

    def estimate_jacobian(self, state: np.ndarray) -> np.ndarray:
        # Mock implementation returning a random jacobian matrix
        return np.random.rand(len(state), len(state))

    def calculate_von_neumann_entropy(self, state: np.ndarray) -> float:
        # Simple Von Neumann entropy calculation based on normalized state
        rho = state / np.sum(state)
        return -np.sum(rho * np.log(rho + 1e-10))

    async def get_current_phi(self) -> float:
        # Mock implementation returning a simulated phi value
        return 0.78 + (np.random.rand() * 0.05)

    async def get_prince_signature(self, level: str, reason: str) -> str:
        # Mock implementation returning a simulated signature
        return "signed-by-prince-" + level

    async def execute_karnak_protocol(self, level: str, reason: str, signature: str):
        print(f"[KARNAK] Executing {level} due to: {reason}. Signature: {signature}")

if __name__ == "__main__":
    monitor = VajraEntropyMonitor()
    asyncio.run(monitor.monitor_system())
