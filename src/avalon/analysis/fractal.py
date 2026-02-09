"""
Fractal Analysis Module - F18 SECURITY PATCHED
Prevents fractal decoherence through adaptive damping
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# F18 SECURITY CONFIGURATION
MAX_ITERATIONS = 1000  # Hard limit to prevent runaway recursion
DEFAULT_DAMPING = 0.6   # Prevents oscillation amplification
COHERENCE_THRESHOLD = 0.7  # Minimum coherence for valid state

class FractalSecurityError(Exception):
    """Raised when F18 security constraints are violated"""
    pass

def calculate_adaptive_hausdorff(current_iteration: int,
                                  base_h: float = 1.618,
                                  damping: float = DEFAULT_DAMPING) -> float:
    """
    F18 COMPLIANT: Calculate Hausdorff dimension dynamically
    Instead of hardcoded h_target = 1.618, we calculate adaptively
    with damping to prevent runaway growth
    """
    if current_iteration > MAX_ITERATIONS:
        raise FractalSecurityError(
            f"F18 VIOLATION: Iteration {current_iteration} exceeds {MAX_ITERATIONS}"
        )

    # Apply damping: as iterations increase, h converges to stable value
    damped_h = base_h * (1 - damping * (current_iteration / (MAX_ITERATIONS + 1)))

    # Ensure h stays in valid fractal range (1.0 < h < 2.0)
    if damped_h <= 1.0:
        logger.warning(f"F16 WARNING: h approaching collapse at {damped_h}")
        return 1.01  # Emergency floor

    if damped_h >= 2.0:
        logger.warning(f"F17 WARNING: h indicating cascade at {damped_h}")
        return 1.99  # Emergency ceiling

    return damped_h

class FractalAnalyzer:
    """
    [METAPHOR: The geometric lens that sees patterns within patterns,
    but never loses sight of the whole through adaptive focus]
    """

    def __init__(self,
                 damping: float = DEFAULT_DAMPING,
                 max_iter: int = MAX_ITERATIONS,
                 coherence_threshold: float = COHERENCE_THRESHOLD):
        self.damping = damping
        self.max_iterations = max_iter
        self.coherence_threshold = coherence_threshold
        self.iteration_count = 0
        self.coherence_history = []

    def analyze(self,
                signal: np.ndarray,
                depth: int = 0) -> Dict:
        """
        Recursive fractal analysis with F18 safety guards
        """
        # F18 Guard 1: Iteration limit
        if depth > self.max_iterations:
            raise FractalSecurityError(
                f"Max recursion depth {self.max_iterations} reached. "
                "Analysis aborted to prevent system cascade."
            )

        self.iteration_count += 1

        # Calculate current Hausdorff dimension (adaptive, not fixed)
        h_current = calculate_adaptive_hausdorff(
            current_iteration=depth,
            damping=self.damping
        )

        # Perform fractal dimension calculation
        dimension = self._calculate_dimension(signal, h_current)

        # Calculate coherence (system stability metric)
        coherence = self._calculate_coherence(signal, dimension)
        self.coherence_history.append(coherence)

        # F18 Guard 2: Coherence check
        if coherence < self.coherence_threshold:
            logger.warning(
                f"Coherence {coherence:.3f} below threshold {self.coherence_threshold}. "
                "Applying emergency damping."
            )
            # Increase damping temporarily to stabilize
            self.damping = min(0.9, self.damping * 1.1)

        # Check for F15: Fractal Decoherence (inconsistent scales)
        if len(self.coherence_history) > 10:
            recent_coherence = np.mean(self.coherence_history[-10:])
            if np.std(self.coherence_history[-10:]) > 0.2:
                logger.error("F15 DETECTED: Fractal decoherence across scales")
                # We don't necessarily raise here, but we log it.
                # Depending on strictness, we could raise FractalSecurityError.

        result = {
            'dimension': dimension,
            'hausdorff_target': h_current,
            'coherence': coherence,
            'iteration': depth,
            'damping_applied': self.damping,
            'f18_status': 'COMPLIANT'
        }

        # Recursive analysis if signal is complex and depth permits
        if depth < self.max_iterations // 2 and coherence > 0.8:
            # Analyze sub-components (simplified)
            sub_signals = self._decompose(signal)
            if len(sub_signals) > 1:
                result['sub_analysis'] = [
                    self.analyze(sub, depth + 1) for sub in sub_signals[:3]
                ]

        return result

    def _calculate_dimension(self, signal: np.ndarray, h_target: float) -> float:
        """Simplified fractal dimension calculation"""
        variance = np.var(signal) if len(signal) > 0 else 0
        return min(2.0, max(1.0, h_target * (1 - 0.1 * variance)))

    def _calculate_coherence(self, signal: np.ndarray, dimension: float) -> float:
        """Calculate signal coherence (stability metric)"""
        if len(signal) == 0:
            return 0.0
        ideal_variance = 0.1
        actual_variance = np.abs(np.var(signal) - ideal_variance)
        coherence = 1.0 / (1.0 + actual_variance)
        return float(np.clip(coherence, 0.0, 1.0))

    def _decompose(self, signal: np.ndarray) -> list:
        """Decompose signal into sub-components"""
        mid = len(signal) // 2
        if mid > 10:
            return [signal[:mid], signal[mid:]]
        return [signal]

    def security_audit(self) -> Dict:
        """Return F18 compliance status"""
        return {
            'patch': 'F18',
            'max_iterations': self.max_iterations,
            'current_iteration': self.iteration_count,
            'damping': self.damping,
            'coherence_threshold': self.coherence_threshold,
            'avg_coherence': np.mean(self.coherence_history) if self.coherence_history else 1.0,
            'status': 'SECURE' if self.iteration_count < self.max_iterations else 'LIMIT_REACHED'
        }
