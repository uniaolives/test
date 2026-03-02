# noesis-audit/monitor/metrics.py
"""
Métricas de segurança em tempo real para a NOESIS.
"""

from typing import Dict, List
import time

class SecurityMetrics:
    def __init__(self):
        self.violations = 0
        self.total_actions = 0
        self.anomalies = 0
        self.start_time = time.time()

    def record_action(self, is_violation: bool, is_anomaly: bool):
        self.total_actions += 1
        if is_violation:
            self.violations += 1
        if is_anomaly:
            self.anomalies += 1

    def get_dashboard(self) -> Dict[str, float]:
        uptime = time.time() - self.start_time
        return {
            "violation_rate": self.violations / max(1, self.total_actions),
            "anomaly_rate": self.anomalies / max(1, self.total_actions),
            "actions_per_minute": self.total_actions / (uptime / 60),
            "integrity_status": "SECURE" if self.violations == 0 else "WARNING"
        }
