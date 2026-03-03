# modules/robotics/hal/python/eeg_toroidal_mapping.py
import numpy as np

def eeg_to_toroidal(alpha, beta, theta, delta, gamma):
    """
    Maps biological brain states from EEG bands to T² coordinates.

    Args:
        alpha, beta, theta, delta, gamma: Power levels of EEG frequency bands.

    Returns:
        (theta_poloidal, phi_toroidal): Coordinates on the torus.
    """
    # theta_poloidal: focus/arousal ratio
    # High beta/gamma (focus) increases theta_poloidal
    theta_poloidal = np.arctan2(beta + gamma, alpha + theta)

    # phi_toroidal: sleep/peak performance ratio
    # High delta (deep sleep) increases phi_toroidal relative to gamma (peak performance)
    phi_toroidal = np.arctan2(delta, gamma + 1e-6) # prevent div by zero

    return (theta_poloidal % (2 * np.pi), phi_toroidal % (2 * np.pi))

def human_swarm_coupling(policy_phi, human_phi, k=0.1):
    """
    Computes the coupling term to phase-lock the swarm to the human operator.

    dphi_swarm/dt = -dH/dphi + k(phi_human - phi_swarm)
    """
    return k * (human_phi - policy_phi)

if __name__ == "__main__":
    # Mock EEG data
    mock_eeg = {
        'alpha': 0.5,
        'beta': 1.2,
        'theta': 0.3,
        'delta': 0.1,
        'gamma': 0.8
    }

    t2_coords = eeg_to_toroidal(**mock_eeg)
    print(f"Human Brain State mapped to T²: theta_poloidal={t2_coords[0]:.2f}, phi_toroidal={t2_coords[1]:.2f}")
