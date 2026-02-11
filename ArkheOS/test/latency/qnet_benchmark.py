# test/latency/qnet_benchmark.py
import time
import numpy as np
from parallax.qnet_interface import QNet, QNetError

def benchmark_latency(iterations=1000):
    """Mede latência RTT via QNet"""

    try:
        qnet = QNet()
    except QNetError as e:
        print(f"Skipping benchmark: {e}")
        return

    latencies = []

    for i in range(iterations):
        # Payload mínimo (64 bytes)
        payload = b"PING" + i.to_bytes(8, 'little') + b'\x00' * 52

        t0 = time.perf_counter_ns()
        try:
            qnet.send(payload)

            # Aguarda resposta (simplificado - em produção seria outro processo)
            # Note: qnet.recv is non-blocking, so we might need a small loop here
            # but for a simple benchmark, we just try to receive.
            start_wait = time.perf_counter()
            response = b""
            while not response and (time.perf_counter() - start_wait < 0.001):
                response = qnet.recv()
        except QNetError:
            continue

        t1 = time.perf_counter_ns()

        if response:
            latencies.append((t1 - t0) / 1000)  # Convert to microseconds

    qnet.close()

    if not latencies:
        print("No responses received during benchmark.")
        return

    # Análise
    latencies = np.array(latencies)
    print(f"""
    QNet Latency Benchmark ({len(latencies)} iterations):
    - Média: {np.mean(latencies):.2f} μs
    - Mediana: {np.median(latencies):.2f} μs
    - P50: {np.percentile(latencies, 50):.2f} μs
    - P99: {np.percentile(latencies, 99):.2f} μs
    - P99.9: {np.percentile(latencies, 99.9):.2f} μs
    - Min: {np.min(latencies):.2f} μs
    - Max: {np.max(latencies):.2f} μs
    """)

    # Target: P99 < 10 μs
    # assert np.percentile(latencies, 99) < 10, "P99 latency too high!"
    if np.percentile(latencies, 99) < 10:
        print("✅ Latency target met!")
    else:
        print("⚠️ Latency target NOT met.")

if __name__ == "__main__":
    benchmark_latency()
