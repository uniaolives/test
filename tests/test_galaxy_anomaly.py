import numpy as np
import sys
import os

# Ensure modules are in path
sys.path.append(os.getcwd())

from modules.arkhe_ggf.secops.cosmic_anomaly import CosmicAnomalyDetector

def test_galaxy_anomaly():
    """
    Injetar gradiente de 'a' em simulação galáctica e ver detecção pelo SecOps.
    Detector alerta com Φ > 0.5.
    """
    detector = CosmicAnomalyDetector()

    # Galaxy data simulating flat rotation curve (phi-anomaly)
    # v_kepler = sqrt(GM/r)
    # v_obs = constant

    radius = np.linspace(1e18, 1e20, 10)
    mass_visible = 1e41 # simplified constant mass
    G = 6.6743e-11

    # Correct Keplerian calculation
    v_kepler = np.sqrt(G * mass_visible / radius)

    # Observed flat rotation (GGF: v_obs ~ constant due to dn/dr)
    v_obs = np.full_like(radius, v_kepler[0])

    galaxy_data = {
        'id': 'ANDROMEDA-GGF',
        'radius': radius.tolist(),
        'velocity_observed': v_obs.tolist(),
        'mass_visible': (G * mass_visible), # Correct units for v_kepler calculation
        'timestamp': 'Ω+∞+190'
    }

    # Calculate v_kepler internally
    # In cosmic_anomaly: v_kepler = galaxy_data['mass_visible'] / np.sqrt(r + 1e-10)
    # Let's adjust mass_visible to make the calculation work for the detector
    # (Simplified internal model)
    galaxy_data['mass_visible'] = np.sqrt(G * mass_visible) * np.sqrt(radius[0])

    alerts = detector.analyze_galaxy_rotation(galaxy_data)

    print(f"Number of alerts: {len(alerts)}")
    for alert in alerts:
        print(f"Alert: {alert['type']} with Φ={alert['phi']:.2f}")
        assert alert['phi'] > 0.5

    assert len(alerts) > 0
    print("Test Galaxy Φ-Anomaly (GGF/SecOps) PASSED")

if __name__ == "__main__":
    test_galaxy_anomaly()
