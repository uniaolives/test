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
    Calculates the hyperbolic distance in the Upper Half-Plane (2D) or Half-Space (3D) model.
    dH(p1, p2) = arcosh(1 + (||p1-p2||^2) / (2 * z1 * z2))
    where z is the vertical coordinate (y in 2D, z in 3D).
    """
    p1 = np.array(p1)
    p2 = np.array(p2)

    # Identify vertical coordinate
    z1 = p1[-1]
    z2 = p2[-1]

    # Ensure z > 0
    z1 = max(0.001, z1)
    z2 = max(0.001, z2)

    sq_dist = np.sum((p1 - p2)**2)
    val = 1 + sq_dist / (2 * z1 * z2)
    return np.arccosh(max(1.0, val))

def atmospheric_density(z, rho0=1.225, H=8500.0):
    """
    Calculates air density at altitude z using the exponential model ρ(z) = ρ0 * exp(-z/H).
    z in meters, H (scale height) approx 8.5km.
    """
    return rho0 * np.exp(-z / H)

def stability_threshold_q_process(d):
    """
    Returns the stability threshold for the Q-process: (d-1)^2 / 8.
    For d=2 (2D): 0.125
    For d=3 (3D): 0.5
    """
    return ((d - 1)**2) / 8.0

def check_q_process_condition(v_max, n_neighbors, d=3):
    """
    Checks the condition for global coherence (Q-process).
    V_max * n_neighbors < threshold
    """
    threshold = stability_threshold_q_process(d)
    return v_max * n_neighbors < threshold

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
