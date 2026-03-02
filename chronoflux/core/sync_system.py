"""
High-precision Timestamping and Synchronization Model.
Replaces the Schumann reference with an Atomic Clock on Chip (CSAC) architecture.
"""
import numpy as np

class AtomicSyncSystem:
    """
    Simulates a network of T.E.N.S.O.R. nodes synchronized via CSAC.
    Enables sub-nanosecond timestamp correlation.
    """
    def __init__(self, node_count=12, stability_ppm=1e-6):
        self.node_count = node_count
        self.stability = stability_ppm # 1 part per million = 1e-6
        self.base_time = 0.0

        # Internal drifts for each node
        self.node_drifts = np.random.normal(0, stability_ppm, node_count)

    def generate_timestamps(self, duration=3600, interval=1.0):
        """
        Generates correlated timestamps across the network.
        """
        steps = int(duration / interval)
        times = np.arange(0, duration, interval)

        node_times = []
        for i in range(self.node_count):
            # True time + drift * true time + jitter
            drifted_time = times * (1 + self.node_drifts[i])
            jitter = np.random.normal(0, 1e-9, len(times)) # 1ns jitter
            node_times.append(drifted_time + jitter)

        return times, np.array(node_times)

    def calculate_sync_error(self, node_times):
        """
        Calculates the relative synchronization error between nodes.
        """
        # Max difference between any two nodes at each step
        errors = np.max(node_times, axis=0) - np.min(node_times, axis=0)
        return errors

if __name__ == "__main__":
    sync = AtomicSyncSystem(stability_ppm=1e-12) # High stability CSAC
    t, nt = sync.generate_timestamps(duration=10)
    err = sync.calculate_sync_error(nt)
    print(f"Max sync error after 10s: {np.max(err)*1e9:.2f} nanoseconds")
