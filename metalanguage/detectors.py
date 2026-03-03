"""
Hierarchical Detector Stack for ANL
Implements statistical and neural-inspired detection levels.
"""

import numpy as np
from typing import List, Dict, Any

class StatisticalDetector:
    """Level 1: unigram entropy, n-gram perplexity."""
    def __init__(self, ngram_order: int = 5):
        self.ngram_order = ngram_order

    def detect(self, probs_q: np.ndarray, probs_p: np.ndarray) -> float:
        """Measure KL divergence as a detection signal."""
        # Avoid zeros
        q = np.where(probs_q == 0, 1e-12, probs_q)
        p = np.where(probs_p == 0, 1e-12, probs_p)
        return np.sum(q * np.log(q / p))

class NeuralLinearDetector:
    """Level 2: Simplified neural embedding classifier."""
    def __init__(self):
        # Simulated weights for detection
        self.weights = np.random.randn(128)

    def detect(self, embedding: np.ndarray) -> float:
        """Return detection probability [0, 1]."""
        # Sigmoid of dot product
        z = np.dot(embedding, self.weights[:len(embedding)])
        return 1.0 / (1.0 + np.exp(-z))

class DetectionHierarchy:
    def __init__(self):
        self.detectors = {
            'statistical': StatisticalDetector(),
            'neural_linear': NeuralLinearDetector()
        }

    def evaluate(self, data: Dict[str, Any]) -> Dict[str, float]:
        results = {}
        if 'probs_q' in data and 'probs_p' in data:
            results['statistical'] = self.detectors['statistical'].detect(data['probs_q'], data['probs_p'])

        if 'embedding' in data:
            results['neural_linear'] = self.detectors['neural_linear'].detect(data['embedding'])

        return results
