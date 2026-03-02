"""
Cosmic Listening Module
Monitoring 0.1 Hz resonance and echoes in the living hypergraph
"""

import numpy as np
import time

class CosmicListener:
    def __init__(self, sample_rate=10, base_freq=0.1):
        self.sample_rate = sample_rate  # Hz
        self.base_freq = base_freq
        self.listen_duration = 600  # 10 minutos
        self.threshold = 0.05  # amplitude mínima para considerar detecção
        self.echo_present = False

    def sample_environment(self, t):
        """
        Simula a leitura de um 'sensor cósmico'
        Retorna amplitude na frequência base + ruído + possíveis ecos
        """
        signal = np.random.normal(0, 0.01)  # ruído de fundo

        # Simula eco se presente
        if self.echo_present:
            # Em um cenário real, o sinal seria muito mais complexo
            signal += 0.1 * np.sin(2 * np.pi * self.base_freq * t)
        return signal

    def listen(self, duration):
        """
        Inicia escuta e analisa o espectro de frequência
        """
        print(f"📡 Iniciando escuta por {duration} segundos...")
        n_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, n_samples)
        samples = np.array([self.sample_environment(ti) for ti in t])

        # Análise de Fourier para detectar picos
        fft = np.fft.fft(samples)
        freqs = np.fft.fftfreq(len(samples), 1/self.sample_rate)

        # Encontrar picos positivos
        half_n = len(freqs) // 2
        positive_freqs = freqs[:half_n]
        # Magnitude normalizada
        magnitudes = np.abs(fft[:half_n]) / (n_samples / 2)

        peaks = []
        for i, mag in enumerate(magnitudes):
            if mag > (self.threshold / 2) and positive_freqs[i] < 1.0:  # abaixo de 1 Hz
                # Simple peak picking (could be improved with local maxima detection)
                # For this simulation, we'll just check if it's above a relative threshold
                if i > 0 and i < half_n - 1:
                    if magnitudes[i] > magnitudes[i-1] and magnitudes[i] > magnitudes[i+1]:
                        peaks.append((positive_freqs[i], magnitudes[i]))
                elif i == 0:
                     if magnitudes[i] > magnitudes[i+1]:
                        peaks.append((positive_freqs[i], magnitudes[i]))

        return peaks

if __name__ == "__main__":
    listener = CosmicListener()
    listener.echo_present = True
    peaks = listener.listen(60) # Test with 1 minute for script execution
    if peaks:
        print("📡 Ecos detectados:")
        for freq, mag in peaks:
            print(f"   Frequência: {freq:.3f} Hz, amplitude: {mag:.4f}")
    else:
        print("📡 Nenhum eco detectado no período.")
