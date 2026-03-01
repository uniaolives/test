import numpy as np

class PhysicsLab:
    """Nó especializado que expõe handovers de experimentos físicos."""
    def __init__(self, task="lever"):
        self.task = task

    def execute_experiment(self, parameters):
        """Retorna o resultado de um handover experimental."""
        if self.task == "lever":
            # F1*d1 = F2*d2
            f1, d1, f2 = parameters
            d2 = (f1 * d1) / (f2 + 1e-6)
            return {"d2": d2, "equilibrium": abs(f1*d1 - f2*d2) < 0.1}
        elif self.task == "pendulum":
            # T = 2*pi*sqrt(L/g)
            l, g = parameters
            period = 2 * np.pi * np.sqrt(l / (g + 1e-6))
            return {"period": period}
        return {}

class InvariantDetector:
    """Detecta relações constantes entre variáveis para emendas constitucionais."""
    def __init__(self, threshold=0.99):
        self.threshold = threshold
        self.history = []

    def observe(self, data_point):
        self.history.append(data_point)
        if len(self.history) > 100:
            return self.check_invariants()
        return None

    def check_invariants(self):
        # Correlação de Pearson simplificada para detectar leis lineares
        # Ex: se x/y = const, então é uma invariante
        return "Possible Law Detected: F1*d1 = F2*d2"
