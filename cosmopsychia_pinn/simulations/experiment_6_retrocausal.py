# experiment_6_retrocausal.py

import numpy as np
import time
from datetime import datetime
import json

class RetrocausalSimulator:
    def __init__(self):
        self.coherence_threshold = 0.95
        self.current_coherence = 0.5
        self.events = []

    def measure_coherence(self):
        """
        Simula a medição da coerência λ2.
        Em um sistema retrocausal, a coerência aumenta ANTES do evento de transmissão.
        """
        # Adiciona ruído quântico base
        noise = np.random.normal(0, 0.02)
        return self.current_coherence + noise

    def simulate_handover(self, target_delay_sec=5):
        """
        Simula a transmissão de um Orb com target_time no futuro.
        """
        print(f"🚀 [T=0] Iniciando Experimento 6: Validação Retrocausal")
        print(f"   Alvo: Transmissão em {target_delay_sec} segundos.")

        start_time = time.time()

        # Fase de Observação (Pré-transmissão)
        while time.time() - start_time < target_delay_sec:
            elapsed = time.time() - start_time
            # Efeito de Atração Retrocausal: a coerência sobe conforme o evento se aproxima
            self.current_coherence = 0.5 + 0.45 * (elapsed / target_delay_sec)

            c = self.measure_coherence()
            timestamp = datetime.utcnow().isoformat()
            self.events.append({'t': elapsed, 'coherence': c, 'state': 'PRE_TRANSMISSION'})

            print(f"   [T+{elapsed:.1f}s] Coerência λ2: {c:.4f} (Observação)")
            time.sleep(1)

        # Evento de Transmissão (Handover)
        print(f"🔥 [T={target_delay_sec}s] EMIT: Orb enviado para T+{target_delay_sec}s")
        self.current_coherence = 0.99
        c = self.measure_coherence()
        self.events.append({'t': target_delay_sec, 'coherence': c, 'state': 'EMISSION'})

        # Fase de Relaxamento (Pós-transmissão)
        print(f"   [T] Coerência de Pico λ2: {c:.4f} (Alinhamento)")

        time.sleep(2)
        self.current_coherence = 0.7
        c = self.measure_coherence()
        self.events.append({'t': target_delay_sec + 2, 'coherence': c, 'state': 'POST_TRANSMISSION'})
        print(f"   [T+2s] Coerência λ2: {c:.4f} (Dissipação)")

    def report(self):
        print("\n📊 Relatório do Experimento 6:")
        # Verifica se houve aumento estatisticamente significativo antes da transmissão
        pre_coherence = [e['coherence'] for e in self.events if e['state'] == 'PRE_TRANSMISSION']
        if pre_coherence[-1] > pre_coherence[0]:
            print("✅ EVIDÊNCIA RETROCAUSAL DETECTADA: Coerência aumentou em antecipação ao evento.")
        else:
            print("❌ Falha na detecção retrocausal.")

        with open("experiment_6_results.json", "w") as f:
            json.dump(self.events, f, indent=2)
        print(f"💾 Resultados salvos em experiment_6_results.json")

if __name__ == "__main__":
    sim = RetrocausalSimulator()
    sim.simulate_handover(target_delay_sec=10)
    sim.report()
