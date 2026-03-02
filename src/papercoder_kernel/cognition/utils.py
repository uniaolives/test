# src/papercoder_kernel/cognition/utils.py
from typing import Any, Tuple

class Memory:
    """Simple experiential memory."""
    def __init__(self):
        self.storage = []

    def store(self, delta: Any):
        self.storage.append(delta)

    def retrieve(self):
        return self.storage

class Environment:
    """Mock environment for evaluation."""
    def __init__(self, target: Any = "success"):
        self.target = target

    def evaluate(self, y: Any) -> Tuple[Any, float]:
        """Evaluates output y and returns feedback and reward."""
        if "y2" in str(y): # Simulate that refinement is better
            return "Good improvement", 0.9
        elif "y1" in str(y):
            return "Basic attempt", 0.3
        else:
            return "Unknown", 0.0
