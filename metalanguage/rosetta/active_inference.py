import numpy as np

def calculate_free_energy(beliefs, likelihood, prior):
    """
    F = D_KL(q(s)||p(s)) - E_q[log p(o|s)]
    Simplified version for ANL Rosetta Stone.
    """
    complexity = np.sum(beliefs * np.log(beliefs / (prior + 1e-12) + 1e-12))
    accuracy = np.sum(beliefs * likelihood)
    return complexity - accuracy

class ActiveInferenceAgent:
    def __init__(self, n_states):
        self.beliefs = np.ones(n_states) / n_states
        self.prior = np.ones(n_states) / n_states

    def update(self, likelihood_vector):
        # Minimize free energy via Bayesian update
        self.beliefs = self.beliefs * likelihood_vector
        self.beliefs /= np.sum(self.beliefs)
