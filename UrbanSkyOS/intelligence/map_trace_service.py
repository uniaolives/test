"""
UrbanSkyOS MapTrace Service
Loads realistic urban trajectories from the google/MapTrace dataset.
Provides paths for drone missions based on real-world map traversals.
"""

import ast
import random
import numpy as np
from datasets import load_dataset

class MapTraceService:
    def __init__(self, dataset_name="google/MapTrace"):
        self.dataset_name = dataset_name
        self._dataset = None
        self._iterator = None

    def _initialize_dataset(self):
        if self._dataset is None:
            try:
                # Load in streaming mode to avoid massive downloads
                self._dataset = load_dataset(self.dataset_name, split='train', streaming=True)
                self._iterator = iter(self._dataset)
            except Exception as e:
                print(f"⚠️ MapTrace: Failed to load dataset {self.dataset_name}: {e}")
                self._dataset = None

    def get_random_path(self):
        """
        Retrieves a random path from the dataset.
        Returns: List of (x, y) normalized coordinates.
        """
        self._initialize_dataset()
        if self._iterator is None:
            # Fallback mock path if dataset unavailable
            return [(0.1, 0.1), (0.2, 0.2), (0.3, 0.1), (0.4, 0.3)]

        try:
            sample = next(self._iterator)
            path_str = sample.get('label', '[]')
            # The label is a string representation of a list of tuples
            path = ast.literal_eval(path_str)
            return path
        except StopIteration:
            self._iterator = iter(self._dataset) # Reset
            return self.get_random_path()
        except Exception as e:
            print(f"⚠️ MapTrace: Error parsing path: {e}")
            return [(0.5, 0.5), (0.6, 0.6)]

    def get_scaled_path(self, scale_factor=100.0, offset=(0, 0)):
        """
        Gets a path and scales it to simulation units.
        """
        path = self.get_random_path()
        scaled_path = []
        for x, y in path:
            scaled_path.append((
                x * scale_factor + offset[0],
                y * scale_factor + offset[1],
                10.0 # Standard altitude
            ))
        return np.array(scaled_path)

if __name__ == "__main__":
    mts = MapTraceService()
    path = mts.get_scaled_path(scale_factor=200)
    print(f"Retrieved scaled path with {len(path)} points.")
    print(f"Start: {path[0]}, End: {path[-1]}")
