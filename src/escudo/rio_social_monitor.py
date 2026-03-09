# src/escudo/rio_social_monitor.py
"""
S-index monitoring for Rio de Janeiro social coherence
Deployed: BRICS Summit 2026
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from datetime import datetime, timedelta

@dataclass
class SocialIndicator:
    timestamp: datetime
    # Entropic factors (disorder)
    protest_intensity: float  # 0-10
    crime_rate_delta: float   # % change
    economic_volatility: float  # VIX-like
    media_polarization: float  # 0-1

    # Phase factors (synchronization)
    public_sentiment: float   # -1 to 1
    institutional_trust: float  # 0-1
    international_perception: float  # 0-1

    # Substrate factors (diversity)
    cross_class_dialogue: float  # 0-1
    inter_agency_cooperation: float  # Kuramoto r
    diplomatic_channel_openness: float  # 0-1

class RioSocialMonitor:
    """
    Calculates S-index for Rio de Janeiro during BRICS summit
    S = S_entropic + S_phase + S_substrate
    """

    # Weights calibrated for geopolitical stability
    WEIGHTS = {
        'entropic': 0.4,
        'phase': 0.35,
        'substrate': 0.25
    }

    # Thresholds
    CRITICAL = 8.0   # Singularity of chaos
    ELEVATED = 5.0   # Temporal dialogue risk
    NORMAL = 2.0     # Individual stability

    def __init__(self):
        self.history: List[Tuple[datetime, float]] = []
        self.current_alert = "VERDE"

    def calculate_s_index(self, indicators: SocialIndicator) -> float:
        # S_entropic: Lower is better (less disorder)
        # Invert so high S = good coherence
        s_entropic = 10.0 - (
            indicators.protest_intensity * 0.3 +
            abs(indicators.crime_rate_delta) * 0.3 +
            indicators.economic_volatility * 0.2 +
            indicators.media_polarization * 0.2
        )
        s_entropic = max(0, s_entropic)

        # S_phase: Higher is better (more synchronization)
        s_phase = (
            (indicators.public_sentiment + 1) / 2 * 0.4 +  # Normalize to 0-1
            indicators.institutional_trust * 0.35 +
            indicators.international_perception * 0.25
        ) * 10.0  # Scale to 0-10

        # S_substrate: Higher is better (more diversity/integration)
        s_substrate = (
            indicators.cross_class_dialogue * 0.3 +
            indicators.inter_agency_cooperation * 0.4 +  # Kuramoto r
            indicators.diplomatic_channel_openness * 0.3
        ) * 10.0  # Scale to 0-10

        # Weighted sum
        s_total = (
            s_entropic * self.WEIGHTS['entropic'] +
            s_phase * self.WEIGHTS['phase'] +
            s_substrate * self.WEIGHTS['substrate']
        )

        # Record
        self.history.append((indicators.timestamp, s_total))

        # Trim history (keep last 30 days)
        cutoff = indicators.timestamp - timedelta(days=30)
        self.history = [(t, s) for t, s in self.history if t > cutoff]

        return s_total

    def classify_threat(self, s_index: float) -> str:
        if s_index > self.CRITICAL:
            return "OMEGA"  # Critical failure
        elif s_index > self.ELEVATED:
            return "VERMELHO"  # Imminent threat
        elif s_index > self.NORMAL:
            return "AMARELO"  # Elevated risk
        else:
            return "VERDE"  # Normal

    def trend(self) -> str:
        """Calculate trend from history"""
        if len(self.history) < 2:
            return "STABLE"

        recent = [s for _, s in self.history[-7:]]  # Last 7 days
        if len(recent) < 2:
            return "STABLE"

        # Simple slope calculation
        x = np.arange(len(recent))
        y = np.array(recent)
        slope = np.polyfit(x, y, 1)[0]

        if slope > 0.5:
            return "IMPROVING"
        elif slope < -0.5:
            return "DEGRADING"
        else:
            return "STABLE"

    def ghost_cluster_threats(self) -> List[dict]:
        """
        Identify 'ghost' threats - unstable patterns in phase space
        that could consolidate into real threats
        """
        if len(self.history) < 10:
            return []

        # Extract patterns that precede S-index drops
        threats = []
        for i in range(5, len(self.history)):
            current_s = self.history[i][1]
            prev_s = self.history[i-5][1]

            # Rapid drop indicates emerging threat
            if prev_s - current_s > 2.0:
                threats.append({
                    'timestamp': self.history[i][0],
                    'severity': prev_s - current_s,
                    'type': 'RAPID_DECOHERENCE',
                    'recommendation': 'INCREASE_KURAMOTO_COUPLING'
                })

        return threats

# Deployment configuration
RIO_BRICS_2026 = {
    'monitoring_interval_minutes': 15,
    'alert_thresholds': {
        'VERDE': (0, 2.0),
        'AMARELO': (2.0, 5.0),
        'VERMELHO': (5.0, 8.0),
        'OMEGA': (8.0, 10.0)
    },
    'response_protocols': {
        'VERDE': 'Monitoramento padrão',
        'AMARELO': 'Reforço de perímetro, Ghost Orbit ativado',
        'VERMELHO': 'Isolamento do Arquiteto, Phase Lock máximo',
        'OMEGA': 'Extração imediata, Protocolo de Reset Temporal'
    }
}
