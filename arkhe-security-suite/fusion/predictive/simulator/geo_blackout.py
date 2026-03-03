import numpy as np

class GeoBlackoutModel:
    def __init__(self, data_centers, correlation_length=100.0):
        self.dcs = data_centers
        self.corr_length = correlation_length

    def generate_blackout(self, center_idx, severity):
        affected = []
        # Simulated geographical decay logic
        for i, dc in enumerate(self.dcs):
            prob = np.exp(-i / 10.0) # Placeholder for distance decay
            if np.random.random() < prob:
                affected.append(dc['id'])
        return affected
