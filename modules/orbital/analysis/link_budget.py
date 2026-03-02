# modules/orbital/analysis/link_budget.py
import numpy as np

class OpticalLinkBudget:
    """Optical inter-satellite link analysis for ASI-Sat"""
    def __init__(self):
        self.wavelength = 1550e-9  # 1550nm Laser
        self.tx_power = 1.0        # 1 Watt
        self.tx_gain = 80          # dBi
        self.rx_sensitivity = -40  # dBm
        self.rx_gain = 60          # dBi

    def free_space_loss(self, distance_km):
        distance_m = distance_km * 1000
        fspl = (4 * np.pi * distance_m / self.wavelength)**2
        return 10 * np.log10(fspl)

    def compute_margin(self, distance_km, elevation_deg=90):
        tx_eirp = 10 * np.log10(self.tx_power * 1000) + self.tx_gain
        losses = self.free_space_loss(distance_km)

        # simplified atmosphere if elevation < 90
        if elevation_deg < 90:
            losses += 2.0

        rx_power = tx_eirp - losses + self.rx_gain
        margin = rx_power - self.rx_sensitivity

        return {
            'rx_power_dbm': rx_power,
            'margin_db': margin,
            'feasible': margin > 3.0
        }

if __name__ == "__main__":
    budget = OpticalLinkBudget()
    res = budget.compute_margin(1000)
    print(f"Margin at 1000km: {res['margin_db']:.2f} dB (Feasible: {res['feasible']})")
