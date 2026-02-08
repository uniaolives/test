"""
F18_Safety_Guard.py - Middleware logic for complexity control
"""

def safety_check(h_val: float) -> float:
    """
    Ensures that the fractal complexity (h) remains within the stability zone.
    Prevent F15 (Fractal Decoherence), F16 (Dimensional Collapse), and F17 (Unfolding Cascade).
    """
    if h_val >= 1.8:
        # Emergency Dampening - Prevent F17
        return 1.44
    if h_val <= 1.2:
        # Complexity Injection - Prevent F16
        return 1.618
    return h_val
