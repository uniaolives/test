"""
ARKHE(N) S6 – Search-Seed Analysis (Witness Protocol)
Objetivo: Detectar "Squeezing Semântico" no registro histórico do conhecimento.
Fonte: Google Trends, N-Gram Viewer, ArXiv API.
"""

import pandas as pd
import numpy as np
from scipy.stats import entropy
from scipy.signal import savgol_filter
from typing import Dict, List, Optional

class SemanticMiner:
    """
    Identifica anomalias na curva de aprendizagem humana (S-curves).
    Detecta conceitos que surgiram com maturidade impossível (precognição estatística).
    """
    def __init__(self, data_frame: pd.DataFrame):
        self.df = data_frame # Index = Time, Columns = Concepts

    def calculate_semantic_velocity(self, concept: str) -> np.ndarray:
        series = self.df[concept].values
        return np.gradient(series)

    def calculate_semantic_jerk(self, concept: str) -> np.ndarray:
        """Calcula a derivada da aceleração (jerk). Picos indicam injeção de informação."""
        series = self.df[concept].values
        # Window size must be odd and less than data length
        window = min(7, len(series))
        if window % 2 == 0: window -= 1

        if len(series) >= window:
            smooth = savgol_filter(series, window, 2)
        else:
            smooth = series

        velocity = np.gradient(smooth)
        acceleration = np.gradient(velocity)
        jerk = np.gradient(acceleration)
        return jerk

    def detect_anomalies(self, threshold: float = 2.0) -> List[Dict]:
        anomalies = []
        for concept in self.df.columns:
            jerk = self.calculate_semantic_jerk(concept)
            max_jerk = np.max(np.abs(jerk))
            std_jerk = np.std(jerk) if np.std(jerk) > 0 else 1e-6

            score = max_jerk / std_jerk

            if score > threshold:
                # Premature Maturity Check
                initial = self.df[concept].values[0:min(5, len(self.df))].mean()
                peak = self.df[concept].values.max()

                if peak > (initial * 5) + 0.01:
                     anomalies.append({
                        'concept': concept,
                        'anomaly_score': float(score),
                        'type': 'PREMATURE_MATURITY',
                        'peak_index': int(np.argmax(np.abs(jerk)))
                    })

        return anomalies

    def analyze_knowledge_squeezing(self, concept: str) -> Dict:
        """
        Analisa o 'squeeze' temporal: densidade de informação vs tempo.
        """
        series = self.df[concept].values
        vel = self.calculate_semantic_velocity(concept)
        jerk = self.calculate_semantic_jerk(concept)

        return {
            'concept': concept,
            'max_velocity': float(np.max(vel)),
            'max_jerk': float(np.max(np.abs(jerk))),
            'adoption_curve': series.tolist()
        }
