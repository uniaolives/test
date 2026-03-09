import time
import numpy as np
from spike_generator import SpikeGenerator
from embedding import NeuralEmbedder

def verify_throughput():
    print("--- Arkhe(n) Neural Interface Throughput Verification ---")
    num_channels = 1024
    duration = 5.0 # seconds
    firing_rate = 50 # Hz
    window_size = 300 # 10ms at 30kHz

    gen = SpikeGenerator(num_channels=num_channels)
    embedder = NeuralEmbedder(dim=num_channels)

    print(f"Simulating {duration}s of data with {num_channels} channels at {firing_rate}Hz...")

    start_gen = time.time()
    all_spikes = gen.generate_spikes(duration, firing_rate_hz=firing_rate)
    end_gen = time.time()

    total_spikes = all_spikes.sum()
    print(f"Generation complete. Total spikes: {total_spikes}")
    print(f"Generation time: {end_gen - start_gen:.4f}s")

    # Process in 10ms windows
    num_windows = all_spikes.shape[1] // window_size
    print(f"Processing {num_windows} windows of 10ms...")

    start_proc = time.time()
    for i in range(num_windows):
        window = all_spikes[:, i*window_size : (i+1)*window_size]
        _ = embedder.embed(window)
    end_proc = time.time()

    total_proc_time = end_proc - start_proc
    throughput = total_spikes / total_proc_time

    print(f"Processing time: {total_proc_time:.4f}s")
    print(f"Throughput: {throughput:.2f} spikes/sec")

    if throughput > 100000:
        print("SUCCESS: Throughput > 100k spikes/sec")
    else:
        print("FAILURE: Throughput targets not met")

if __name__ == "__main__":
    verify_throughput()
