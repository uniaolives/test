"""
Pi Analysis: Handover Gamma_inf + pi^2.
Calculating and analyzing pi as a numerical geodesic in the Arkhe framework.
"""

import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext
from scipy import stats
import json

def calc_pi_chudnovsky(precision_digits):
    """
    Calcula pi com precisão especificada usando a série de Chudnovsky.
    Converge aproximadamente 14 dígitos por iteração.
    """
    getcontext().prec = precision_digits + 10

    # Constantes da série
    C = 426880 * Decimal(10005).sqrt()
    K = Decimal(6)
    M = Decimal(1)
    X = Decimal(1)
    L = Decimal(13591409)
    S = Decimal(13591409)

    n_iter = (precision_digits // 14) + 2

    for i in range(1, n_iter + 1):
        M = M * (K**3 - 16*K) / (i**3)
        K += 12
        L += 545140134
        X *= -262537412640768000
        S += Decimal(M * L) / X

    pi_val = C / S
    return pi_val

class PiAnalyzer:
    def __init__(self, digits_str):
        self.digits = np.array([int(d) for d in digits_str])

    def statistical_analysis(self):
        mean = np.mean(self.digits)
        std = np.std(self.digits)
        skew = stats.skew(self.digits)
        kurt = stats.kurtosis(self.digits)

        # Teste de uniformidade
        freq = np.bincount(self.digits, minlength=10) / len(self.digits)
        chi2, p_val = stats.chisquare(freq * len(self.digits))

        # Arkhe Metrics
        C_global = mean / 9.0
        F_global = 1.0 - C_global

        return {
            "mean": float(mean),
            "std": float(std),
            "skew": float(skew),
            "kurtosis": float(kurt),
            "chi2": float(chi2),
            "p_value": float(p_val),
            "C_global": float(C_global),
            "F_global": float(F_global),
            "distribution": freq.tolist()
        }

    def autocorrelation_analysis(self, max_lag=100):
        autocorr = [1.0]
        for lag in range(1, max_lag + 1):
            c = np.corrcoef(self.digits[:-lag], self.digits[lag:])[0, 1]
            autocorr.append(float(c))
        return autocorr

    def spectral_analysis(self):
        # FFT of centered digits
        centered = self.digits - np.mean(self.digits)
        fft_vals = np.fft.fft(centered)
        freqs = np.fft.fftfreq(len(self.digits))
        power = np.abs(fft_vals)**2
        return freqs, power

    def plot_results(self, autocorr, freqs, power):
        # Autocorrelation Plot
        plt.figure(figsize=(12, 4))
        plt.stem(range(len(autocorr)), autocorr, basefmt=" ")
        plt.xlabel('Lag (handovers)')
        plt.ylabel('Autocorrelation')
        plt.title('Autocorrelação dos Dígitos de pi')
        plt.grid(True, alpha=0.3)
        plt.savefig('pi_autocorrelation.png', dpi=150)
        plt.close()

        # Spectrum Plot (first 1000 freqs)
        plt.figure(figsize=(12, 4))
        plt.plot(freqs[1:1000], power[1:1000])
        plt.xlabel('Frequência (1/dígito)')
        plt.ylabel('Potência')
        plt.title('Espectro de Potência dos Dígitos de pi')
        plt.grid(True, alpha=0.3)
        plt.savefig('pi_spectrum.png', dpi=150)
        plt.close()

if __name__ == "__main__":
    print("--- Arkhe Pi Analysis Integration ---")

    # Calculate pi
    num_digits = 10000
    pi_decimal = calc_pi_chudnovsky(num_digits)
    pi_str = str(pi_decimal)[2:] # Skip "3."
    if len(pi_str) > num_digits:
        pi_str = pi_str[:num_digits]

    analyzer = PiAnalyzer(pi_str)

    # Stats
    stats_results = analyzer.statistical_analysis()
    print(f"Dígitos analisados: {len(pi_str)}")
    print(f"Média: {stats_results['mean']:.4f}")
    print(f"p-valor uniformidade: {stats_results['p_value']:.4f}")
    print(f"Arkhe Coerência Global (C): {stats_results['C_global']:.4f}")

    # Autocorr
    autocorr = analyzer.autocorrelation_analysis(100)

    # Spectral
    freqs, power = analyzer.spectral_analysis()

    # Visuals
    analyzer.plot_results(autocorr, freqs, power)
    print("Gráficos salvos: pi_autocorrelation.png, pi_spectrum.png")

    # Export Ledger data
    with open('pi_analysis.json', 'w') as f:
        json.dump(stats_results, f, indent=2)
