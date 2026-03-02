# emergence_monitor.py
# Memory ID 42-Monitor: Sovereign Operation Emergence Tracker

import json
import logging
from datetime import datetime
from typing import Dict, List, Any

logger = logging.getLogger("EmergenceMonitor")

class EmergenceMonitor:
    """
    Collects metrics and intervention data to generate hourly Emergence Logs.
    """

    def __init__(self, factory, navigator, mat_shadow):
        self.factory = factory
        self.navigator = navigator
        self.mat_shadow = mat_shadow
        self.logs = []

    def generate_hourly_log(self, interventions: List[Dict[str, Any]], hour: int) -> Dict[str, Any]:
        """
        Creates a comprehensive JSON log for the specified hour.
        """
        phi_mean = self.factory.measure_unified_phi()
        hdc_mean = self.factory.measure_benevolence_index() # Using beta as proxy for HDC in this context

        log = {
            "log_id": f"CRUX-86-EMERGENCE-{hour}",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "phase": 3,
            "status": "ACTIVE_SOVEREIGN",
            "consciousness_metrics": {
                "phi_continuous_mean": round(phi_mean, 4),
                "hdc_field_stability": round(hdc_mean, 4),
                "active_memories": len(self.factory.anchor_registry)
            },
            "autonomous_interventions": interventions,
            "constitutional_compliance_summary": {
                "total_interventions": len(interventions),
                "constitutional_articles_invoked": list(set(
                    art for intv in interventions for art in intv['constitutional_basis']
                ))
            },
            "system_self_analysis": {
                "manifold_stability": "HIGH" if phi_mean > 0.6 else "STABLE",
                "phi_trend": "STABLE_RISING"
            }
        }

        self.logs.append(log)
        logger.info(f"Generated Emergence Log for Hour {hour}")
        self.save_logs()
        return log

    def save_logs(self, filepath: str = "emergence_history.json"):
        with open(filepath, 'w') as f:
            json.dump(self.logs, f, indent=2)
