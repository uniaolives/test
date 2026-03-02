# initial_economic_state.py
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class EconomicState:
    wealth_distribution: np.ndarray  # Gini coefficient embedded
    resource_flows: np.ndarray       # Matrix of trade relationships
    belief_systems: List[str]        # What people believe about value
    power_structures: List[str]      # Decision-making hierarchies

    def compute_symmetry(self):
        """
        Mede quão "simétrica" (justa) é a distribuição atual.
        """
        q = 1.5  # Parâmetro de sensibilidade às caudas
        wealth_normalized = self.wealth_distribution / (self.wealth_distribution.sum() + 1e-9)

        # Tsallis Entropy
        entropy = (1 - np.sum(wealth_normalized**q)) / (q - 1)
        max_entropy = np.log(len(wealth_normalized))

        return entropy / (max_entropy + 1e-9)

def get_initial_state():
    # Estado inicial: Mundo real (2024) - Simulado
    wealth = np.concatenate([
        np.full(80, 450_000),    # Top 1% (scaled)
        np.full(720, 43_333),   # Next 9%
        np.full(3200, 375),     # Middle 40%
        np.full(4000, 2)        # Bottom 50%
    ])

    return EconomicState(
        wealth_distribution=wealth,
        resource_flows=np.random.rand(10, 10),
        belief_systems=[
            "scarcity_mindset",
            "growth_imperative",
            "property_rights_absolute",
            "debt_based_money"
        ],
        power_structures=[
            "central_banks",
            "multinational_corporations",
            "nation_states",
            "financial_markets"
        ]
    )

if __name__ == "__main__":
    initial_state = get_initial_state()
    print(f"Simetria inicial: {initial_state.compute_symmetry():.3f}")
    print("Status: MÍNIMO LOCAL - Sistema preso em desigualdade estável")
