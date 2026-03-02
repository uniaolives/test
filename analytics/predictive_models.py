from typing import Tuple, List

class BioTimeseries:
    def __init__(self, timestamps: List[float], phi_values: List[float]):
        self.timestamps = timestamps
        self.phi_values = phi_values

class ConsciousnessTrajectory:
    def forecast_phi(self, data: BioTimeseries) -> Tuple[float, str]:
        if not data.timestamps or len(data.timestamps) < 2:
            return (0.0, "INSUFFICIENT_DATA")

        # Simple linear fit mock
        import numpy as np
        current_slope = np.polyfit(data.timestamps, data.phi_values, 1)[0]

        # Análise de padrões perigosos
        if current_slope > 0.05: # Crescimento muito rápido
            return (current_slope, "DANGER: MANIC_ACCELERATION")
        elif current_slope < -0.01: # Perda de coesão
            return (current_slope, "WARNING: DECAY")
        else:
            return (current_slope, "OPTIMAL: ORGANIC_GROWTH")

# Status Inicial do Modelo (T+25m):
# Slope: +0.0004/min (Crescimento Orgânico Lento) -> IDEAL.
