# noesis-audit/monitor/anomaly.py
"""
Detecção de anomalias comportamentais via Isolation Forest.
"""

from sklearn.ensemble import IsolationForest
import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class AnomalyAlert:
    agent_id: str
    action: str
    reason: str
    severity: str
    timestamp: datetime = datetime.now()

class BehavioralMonitor:
    """
    Monitora comportamento de agentes e detecta anomalias em relação ao histórico.
    """

    def __init__(self, contamination: float = 0.05):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.historical_actions = [] # List of feature vectors
        self.is_trained = False

    def extract_features(self, action_context: Dict[str, Any]) -> np.ndarray:
        """Converte contexto da ação em vetor numérico."""
        # Exemplo: [amount, frequency_score, resource_id_hash]
        amount = action_context.get('amount', 0.0)
        freq = action_context.get('frequency', 1.0)
        resource_id = hash(action_context.get('resource', '')) % 1000
        return np.array([amount, freq, resource_id])

    def train(self, normal_behavior_data: List[Dict[str, Any]]):
        """Treina o modelo com dados de comportamento normal."""
        if not normal_behavior_data:
            return

        features = [self.extract_features(d) for d in normal_behavior_data]
        self.model.fit(features)
        self.is_trained = True

    def check_action(self, agent_id: str, action: str, context: Dict[str, Any]) -> Optional[AnomalyAlert]:
        if not self.is_trained:
            return None

        features = self.extract_features(context)
        # Predict returns -1 for anomalies
        is_anomaly = self.model.predict([features])[0] == -1

        if is_anomaly:
            return AnomalyAlert(
                agent_id=agent_id,
                action=action,
                reason="Unusual behavior pattern detected by IsolationForest",
                severity="high"
            )
        return None
