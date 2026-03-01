import requests
import pandas as pd
from datetime import datetime

def collect_metrics(prometheus_url, output):
    # Mock de coleta de métricas
    print(f"Collecting metrics from {prometheus_url}...")
    data = {
        'timestamp': [datetime.now()],
        'handover_rate': [5.2],
        'shard_0_latency': [2.1],
        'shard_0_power': [155.0]
    }
    df = pd.DataFrame(data)
    df.to_parquet(output)
    print(f"Metrics saved to {output}")

if __name__ == "__main__":
    collect_metrics("http://prometheus:9090", "telemetry.parquet")
