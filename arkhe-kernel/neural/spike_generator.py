import numpy as np
import time

class SpikeGenerator:
    """
    Simulates a synthetic neural spike train using a Poisson process.
    Target frequency: 10Hz - 1kHz per channel.
    """
    def __init__(self, num_channels=1024, sampling_rate=30000):
        self.num_channels = num_channels
        self.sampling_rate = sampling_rate # Hz

    def generate_spikes(self, duration_sec, firing_rate_hz=20):
        """
        Generates binary spike matrix (channels x time_steps)
        """
        num_steps = int(duration_sec * self.sampling_rate)
        prob_spike = firing_rate_hz / self.sampling_rate

        # Using numpy for vectorized generation
        spikes = np.random.rand(self.num_channels, num_steps) < prob_spike
        return spikes

if __name__ == "__main__":
    gen = SpikeGenerator(num_channels=1024)
    start = time.time()
    spikes = gen.generate_spikes(1.0, firing_rate_hz=50)
    end = time.time()

    print(f"Generated {spikes.sum()} spikes across {spikes.shape[0]} channels and {spikes.shape[1]} steps.")
    print(f"Time taken for 1.0s simulation: {end - start:.4f}s")
