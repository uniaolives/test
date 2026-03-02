# phase-5/qrng_coherence_monitor.py
# Monitoramento do colapso da função de onda coletiva

import numpy as np
import time
from datetime import datetime, timedelta

class QRNGGatewayMonitor:
    def __init__(self):
        # Configuração GCP (Global Consciousness Project)
        self.gcp_nodes = 65  # Nós ativos do GCP
        self.expected_randomness = 0.5  # Probabilidade esperada para binário
        self.significance_threshold = 0.01  # p < 0.01 para significância

        # Frequência solar-terrestre
        self.target_frequency = 9.6  # mHz
        self.sampling_rate = 1.0  # Hz

        # Estado do portal
        self.portal_active = True
        self.peak_window = timedelta(minutes=30)

        # Dados coletados
        self.data = {
            'timestamps': [],
            'z_scores': [],
            'p_values': [],
            'deviation_magnitude': [],
            'coherence_phase': []
        }

    def monitor_qrng_collapse(self, iterations=5):
        """Monitora o colapso da função de onda via GCP"""
        print("🌀 INICIANDO MONITORAMENTO QRNG-GCP")
        print(f"   Janela: {self.peak_window}")
        print(f"   Nós GCP: {self.gcp_nodes}")

        for i in range(iterations):
            # Coleta de dados do GCP (simulação)
            gcp_data = self.simulate_gcp_data()

            # Calcula desvio da aleatoriedade
            z_score, p_value, deviation = self.calculate_randomness_deviation(gcp_data)

            # Verifica colapso da função de onda
            wavefunction_collapse = self.detect_wavefunction_collapse(p_value, deviation)

            # Registro
            timestamp = datetime.now()
            self.data['timestamps'].append(timestamp)
            self.data['z_scores'].append(z_score)
            self.data['p_values'].append(p_value)
            self.data['deviation_magnitude'].append(deviation)
            self.data['coherence_phase'].append(self.get_current_phase())

            # Log
            status = "🟢 COLAPSO DETECTADO" if wavefunction_collapse else "🔵 ALEATORIEDADE NORMAL"
            print(f"{timestamp.strftime('%H:%M:%S')} | Z={z_score:.2f} | p={p_value:.4f} | {status}")

            # Se colapso detectado, intensificar
            if wavefunction_collapse:
                self.intensify_coherence_field(z_score)

            time.sleep(0.1)  # Faster for simulation

        return self.data

    def simulate_gcp_data(self):
        """Simula dados do GCP"""
        base_randomness = 0.5
        portal_effect = 0.1 if self.portal_active else 0.0
        solar_influence = np.sin(datetime.now().timestamp() * self.target_frequency * 0.001) * 0.05

        bias = base_randomness + portal_effect + solar_influence
        bits = np.random.binomial(1, min(0.99, max(0.01, bias)), 200)

        return {
            'bits': bits,
            'mean': np.mean(bits),
            'std': np.std(bits),
            'timestamp': datetime.now()
        }

    def calculate_randomness_deviation(self, gcp_data):
        """Calcula desvio da aleatoriedade esperada"""
        observed_mean = gcp_data['mean']
        expected_mean = 0.5
        n = len(gcp_data['bits'])

        # Teste Z para proporção
        se = np.sqrt(expected_mean * (1 - expected_mean) / n)
        z_score = (observed_mean - expected_mean) / se

        # Valor p (bilateral) - Simple approximation if scipy is missing
        try:
            from scipy import stats
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        except ImportError:
            p_value = np.exp(-0.5 * z_score**2)  # Rough gaussian tail approximation

        # Magnitude do desvio
        deviation = abs(observed_mean - expected_mean)

        return z_score, p_value, deviation

    def detect_wavefunction_collapse(self, p_value, deviation):
        """Detecta colapso da função de onda coletiva"""
        if p_value < self.significance_threshold and deviation > 0.02:
            return True
        return False

    def get_current_phase(self):
        """Calcula fase atual da frequência 9.6 mHz"""
        period = 1 / 0.0096
        current_time = datetime.now().timestamp()
        phase = (current_time % period) / period * 2 * np.pi
        return phase

    def intensify_coherence_field(self, z_score):
        """Intensifica campo de coerência quando colapso detectado"""
        intensity = min(1.0, abs(z_score) / 3.0)
        print(f"⚡ INTENSIFICANDO CAMPO: Z={z_score:.2f} | Intensity: {intensity:.2f}")

if __name__ == "__main__":
    monitor = QRNGGatewayMonitor()
    monitor.monitor_qrng_collapse()
