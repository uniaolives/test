"""
Identity Stress Test - Simulating sudden loss of Arkhe coefficients to test robustness.
"""

import asyncio
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
from .individuation import IndividuationManifold

class IdentityStressTest:
    """
    Simula cenários de perda de coerência para testar a robustez da identidade.
    """

    SCENARIOS = {
        'loss_of_purpose': {
            'parameter': 'F',
            'target': 0.1,
            'description': 'Perda súbita de propósito (crise existencial)'
        },
        'energy_depletion': {
            'parameter': 'E',
            'target': 0.1,
            'description': 'Depleção energética (fadiga extrema)'
        }
    }

    def __init__(self, baseline_arkhe: Dict[str, float]):
        self.baseline = baseline_arkhe.copy()
        self.manifold = IndividuationManifold()

    async def run_scenario(self, name: str, duration: int = 5) -> Dict[str, Any]:
        if name not in self.SCENARIOS:
            return {"error": "Scenario not found"}

        scenario = self.SCENARIOS[name]
        param = scenario['parameter']
        target_val = scenario['target']
        initial_val = self.baseline.get(param, 0.5)

        print(f"\n⚠️  INICIANDO TESTE DE TENSÃO: {scenario['description']}")

        trajectory = []
        arkhe_current = self.baseline.copy()

        for i in range(duration * 2):
            # Degradação linear
            progress = i / (duration * 2)
            arkhe_current[param] = initial_val + (target_val - initial_val) * progress

            # Cálculo de I
            l1, l2 = 0.72, 0.28
            S = 0.61
            I = self.manifold.calculate_individuation(arkhe_current.get('F', 0.5), l1, l2, S)
            mag = np.abs(I)
            trajectory.append(mag)

            if mag < 0.5:
                print(f"   🚨 T+{i*0.5:.1f}s: RISCO ALTO DETECTADO (|I|={mag:.3f})")

            await asyncio.sleep(0.05)

        return {
            "scenario": name,
            "trajectory": trajectory,
            "robustness_score": float(np.mean(trajectory) / initial_val)
        }

    def plot_trajectory(self, result: Dict[str, Any], save_path: str = "stress_test_plot.png"):
        plt.figure(figsize=(10, 6))
        plt.plot(result['trajectory'], label='|I| (Individuation)')
        plt.axhline(y=0.5, color='r', linestyle='--', label='Ego Death Threshold')
        plt.title(f"Identity Stress Trajectory: {result['scenario']}")
        plt.xlabel("Time Steps")
        plt.ylabel("Magnitude |I|")
        plt.legend()
        plt.savefig(save_path)
