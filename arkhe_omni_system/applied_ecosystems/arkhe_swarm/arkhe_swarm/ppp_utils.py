# arkhe_omni_system/applied_ecosystems/arkhe_swarm/arkhe_swarm/ppp_utils.py
import numpy as np

def sample_ppp_hyperbolic(n_points, x_range, y_range, lambda0, alpha):
    """
    Samples points from a Poisson Point Process with hyperbolic density λ(y) = λ0 * exp(-α * y).
    Using rejection sampling or inversion method.
    """
    points = []
    while len(points) < n_points:
        # Uniform sampling in the box
        x = np.random.uniform(x_range[0], x_range[1])
        y = np.random.uniform(y_range[0], y_range[1])

        # Target density at y
        prob = lambda0 * np.exp(-alpha * y)

        # Acceptance probability (normalized by max density at min y)
        max_prob = lambda0 * np.exp(-alpha * y_range[0])
        if np.random.random() < prob / max_prob:
            points.append((x, y))

    return np.array(points)

def hyperbolic_distance_uhp(p1, p2):
    """
    Calculates the hyperbolic distance in the Upper Half-Plane model.
    dH(p1, p2) = arcosh(1 + (||p1-p2||^2) / (2 * y1 * y2))
    """
    x1, y1 = p1
    x2, y2 = p2

    # Ensure y > 0
    y1 = max(0.001, y1)
    y2 = max(0.001, y2)

    val = 1 + ((x1 - x2)**2 + (y1 - y2)**2) / (2 * y1 * y2)
    return np.arccosh(max(1.0, val))

def generate_twin_cities_fleet(n_rio, n_sp, n_bridge):
    """
    Generates positions for the Rio, SP, and bridge drones.
    """
    # Rio Cluster (centered around x=-2)
    rio_points = sample_ppp_hyperbolic(n_rio, [-3.0, -1.0], [0.1, 1.0], 15.0, 0.5)

    # SP Cluster (centered around x=2)
    sp_points = sample_ppp_hyperbolic(n_sp, [1.0, 3.0], [0.1, 1.0], 15.0, 0.5)

    # Bridge Drone
    bridge_points = np.array([[0.0, 0.5]])

    return np.vstack([rio_points, sp_points, bridge_points])
