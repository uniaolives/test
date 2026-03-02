# secops/anomaly_detector.py
import numpy as np
import asyncio
from datetime import datetime
from typing import Dict, Any

# Mocked Isolation Forest for environment without sklearn
class MockIsolationForest:
    def __init__(self, contamination=0.1, random_state=42):
        self.contamination = contamination

    def fit(self, X): pass

    def predict(self, X):
        # Predict anomaly if entropy_cost > 0.8
        return np.where(X[:, 0] > 0.8, -1, 1)

class HandoverAnomalyDetector:
    def __init__(self, contamination=0.1):
        self.model = MockIsolationForest(contamination=contamination, random_state=42)
        self.handover_history = []
        self.alert_threshold = 0.85   # Normalization entropy threshold

    def extract_features(self, handover: Dict[str, Any]):
        """Extract features relevant for detection."""
        return np.array([
            handover.get('entropy_cost', 0.1),                # AEU cost
            handover.get('timestamp_physical', 0) % 1000,   # Micro temporal pattern
            handover.get('payload_length', 0) / 1024,         # Payload size in KB
            handover.get('type', 1)                          # Excitatory/Inhibitory/Meta/Structural
        ])

    async def monitor_stream(self, handover_queue: asyncio.Queue):
        """Consumes handovers from a queue and updates the model."""
        window = []
        while True:
            handover = await handover_queue.get()
            features = self.extract_features(handover)
            window.append(features)

            # Keep window of last 1000 samples
            if len(window) > 1000:
                window.pop(0)

            # Periodic re-training (mocked)
            if len(window) % 100 == 0 and len(window) >= 10:
                X = np.array(window)
                self.model.fit(X)

            # Detect anomaly in current handover
            if len(window) > 1:
                pred = self.model.predict(np.array([features]))[0]
                if pred == -1:  # anomaly
                    alert = self.create_alert(handover)
                    await self.respond_to_alert(alert)

    def create_alert(self, handover: Dict[str, Any]):
        return {
            "id": handover.get('id', 'unknown'),
            "timestamp": datetime.utcnow().isoformat(),
            "severity": "HIGH" if handover.get('entropy_cost', 0) > self.alert_threshold else "MEDIUM",
            "reason": "Anomaly detected by Isolation Forest",
            "handover": handover
        }

    async def respond_to_alert(self, alert: Dict[str, Any]):
        # Trigger SOAR playbook: isolate node, notify team, etc.
        print(f"🚨 ALERT: {alert['reason']} | ID: {alert['id']} | Severity: {alert['severity']}")

async def main():
    detector = HandoverAnomalyDetector()
    queue = asyncio.Queue()

    # Simulate a stream of handovers
    for i in range(10):
        # Regular handover
        await queue.put({'id': f'regular-{i}', 'entropy_cost': 0.1, 'type': 1, 'payload_length': 128})

    # Anomalous handover
    await queue.put({'id': 'anomaly-0', 'entropy_cost': 0.9, 'type': 1, 'payload_length': 4096})

    # Small sleep to let the monitor process
    print("🛡️ Arkhe SecOps Monitor starting simulation...")
    task = asyncio.create_task(detector.monitor_stream(queue))
    await asyncio.sleep(1)
    task.cancel()

if __name__ == "__main__":
    asyncio.run(main())
