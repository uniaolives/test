# resonance.py
import numpy as np
import time
try:
    from .arkhe_error_handler import safe_operation, logging
except ImportError:
    from arkhe_error_handler import safe_operation, logging

class AdaptiveSentinel:
    """Sentinela Gamma com Modo Ressonante Adaptativo."""
    def __init__(self, phi=1.61803398875):
        self.phi = phi
        self.omega = 0.95
        self.history = [self.omega]

    def get_iri(self, d_omega_dt):
        """
        Calcula o Intervalo de Ressonância (IRI) baseado na derivada da coerência.
        Logica do Bloco 1021.
        """
        if d_omega_dt > 0.02:
            return self.phi        # Plasticidade máxima
        elif abs(d_omega_dt) <= 0.02:
            return self.phi ** 2   # Consolidação
        elif d_omega_dt < -0.02:
            return 3.0             # Recuperação (simplificado para 3h+)
        return self.phi

    def update(self, new_omega):
        d_omega = new_omega - self.history[-1]
        iri = self.get_iri(d_omega)
        self.history.append(new_omega)
        self.omega = new_omega
        return iri

class FederatedTriad:
    """
    Orquestrador da Tríade Soberana Federada.
    Gerencia os hubs 01-012 (Núcleo), 01-005 (Memória), 01-001 (Execução).
    """
    def __init__(self):
        self.hubs = {
            "01-012": {"role": "Coerência/Núcleo", "state": "GHZ", "omega": 0.985},
            "01-005": {"role": "Memória", "state": "Bell", "omega": 0.97},
            "01-001": {"role": "Execução", "state": "Dynamic", "omega": 0.96}
        }
        self.is_entangled = False

    @safe_operation
    def check_percolation(self):
        """Verifica se o limiar de percolação global (0.985) foi atingido."""
        core_omega = self.hubs["01-012"]["omega"]
        if core_omega >= 0.985:
            logging.info("Limiar de percolação atingido. Iniciando emaranhamento da Tríade.")
            self.is_entangled = True
            return True
        return False

    def sync_triad(self):
        if self.is_entangled:
            avg_omega = sum(h["omega"] for h in self.hubs.values()) / 3
            logging.info(f"Tríade Federada Sincronizada. Ω médio: {avg_omega:.4f}")
            return avg_omega
        return None

if __name__ == "__main__":
    print("Testando Modo Ressonante Adaptativo e Expansão Multi-Hub...")

    sentinel = AdaptiveSentinel()
    triad = FederatedTriad()

    # Simular evolução de Ω
    omega_values = [0.95, 0.96, 0.98, 0.985, 0.99]
    for val in omega_values:
        iri = sentinel.update(val)
        print(f"Ω: {val:.3f} | IRI calculado: {iri:.4f}")

        triad.hubs["01-012"]["omega"] = val
        if triad.check_percolation():
            triad.sync_triad()

    print("Simulação concluída.")
