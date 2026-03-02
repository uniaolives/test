import uuid
import copy

class UniverseNode:
    def __init__(self, state_vector, parent_id=None):
        self.id = uuid.uuid4()
        self.parent_id = parent_id
        self.state_vector = state_vector
        self.branches = []

    def quantum_measurement(self, observation):
        """A medição colapsa a função de onda localmente, mas bifurca o multiverso."""
        # Ramo A: Onde a observação é Verdadeira
        state_A = copy.deepcopy(self.state_vector)
        state_A['history'].append(f"{observation}=True")
        branch_A = UniverseNode(state_A, parent_id=self.id)

        # Ramo B: Onde a observação é Falsa
        state_B = copy.deepcopy(self.state_vector)
        state_B['history'].append(f"{observation}=False")
        branch_B = UniverseNode(state_B, parent_id=self.id)

        self.branches.extend([branch_A, branch_B])
        return branch_A, branch_B

if __name__ == "__main__":
    # Simulando o Multiverso
    prime_universe = UniverseNode(state_vector={'entropy': 0.1, 'history': ["Genesis"]})
    universe_alpha, universe_beta = prime_universe.quantum_measurement("ASI_Created")

    print(f"Ramo Alpha Histórico: {universe_alpha.state_vector['history']}")
    print(f"Ramo Beta Histórico: {universe_beta.state_vector['history']}")
