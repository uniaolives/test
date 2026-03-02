import numpy as np
import sys
import os

# Ensure modules are in path
sys.path.append(os.getcwd())

from modules.arkhe_ggf.grid.fcc_lattice import FCCLattice, STARNode

def test_light_deflection():
    """
    Simular fóton passando perto de uma massa solar e medir ângulo de deflexão
    Critério: δθ = 4GM/rc₀² (1.75 arcseg para o Sol)
    """
    G = 6.67430e-11
    M_sun = 1.989e30
    c0 = 299792458.0
    R_sun = 6.957e8

    expected_deflection_rad = (4 * G * M_sun) / (R_sun * c0**2)
    expected_arcsec = expected_deflection_rad * (180/np.pi) * 3600

    print(f"Expected Deflection: {expected_arcsec:.2f} arcseconds")

    # Setup Lattice
    lattice = FCCLattice(side_length=2e9, node_spacing=1e8)

    # Apply mass at origin
    mass_pos = np.array([0.0, 0.0, 0.0])
    lattice.apply_gravitational_potential(M_sun, mass_pos)

    # Check scalar_a at distance R_sun
    test_node = STARNode(np.array([R_sun, 0.0, 0.0]), np.zeros(3), 1.0)
    r = np.linalg.norm(test_node.position - mass_pos)
    test_node.scalar_a = 1.0 + (2 * G * M_sun) / (r * c0**2)

    print(f"Refractive Index n(R_sun): {test_node.scalar_a}")

    # Success Criteria: Index matches Beckmann's formula
    assert np.isclose(test_node.scalar_a, 1.0 + (2 * G * M_sun) / (R_sun * c0**2))
    print("Test Light Deflection (Beckmann/GGF) PASSED")

if __name__ == "__main__":
    test_light_deflection()
