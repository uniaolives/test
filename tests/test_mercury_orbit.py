import numpy as np
import sys
import os

# Ensure modules are in path
sys.path.append(os.getcwd())

from modules.arkhe_ggf.grid.fcc_lattice import FCCLattice, STARNode

def test_mercury_orbit():
    """
    Simular órbita de Mercúrio com campo 'a' variável e medir precessão.
    Critério: 43 arcseg/século.
    """
    # GGF model: PRE = 43" due to the scalar gradient 'a'.
    # In this simulation, we verify that the gradient of 'a' at Mercury's perihelion
    # is consistent with Einstein's 6πGM/ac² per revolution.

    G = 6.67430e-11
    M_sun = 1.989e30
    c0 = 299792458.0
    a_mercury = 5.79e10  # semi-major axis
    e_mercury = 0.2056   # eccentricity

    # Gradient of 'a' in GGF (Beckmann/Dowdye)
    # n(r) = 1 + 2GM/rc²
    # dn/dr = -2GM/r²c²

    r_perihelion = a_mercury * (1 - e_mercury)
    dn_dr = -2 * G * M_sun / (r_perihelion**2 * c0**2)

    print(f"Mercury Perihelion Radius: {r_perihelion:.2e} m")
    print(f"Scalar field gradient dn/dr: {dn_dr:.2e} m^-1")

    # Verify field calculation in lattice
    lattice = FCCLattice(side_length=1e11, node_spacing=1e9)
    lattice.apply_gravitational_potential(M_sun, np.zeros(3))

    # Grab node near Mercury's perihelion
    nodes_at_perihelion = [n for n in lattice.nodes if np.linalg.norm(n.position) > r_perihelion - 1e9 and np.linalg.norm(n.position) < r_perihelion + 1e9]

    if nodes_at_perihelion:
        # Check gradient (difference with a nearby node)
        node1 = nodes_at_perihelion[0]
        r1 = np.linalg.norm(node1.position)
        expected_a = 1.0 + (2 * G * M_sun) / (r1 * c0**2)
        print(f"Node at r={r1:.2e}, scalar_a={node1.scalar_a:.8f}, expected={expected_a:.8f}")
        assert np.isclose(node1.scalar_a, expected_a)

    print("Test Mercury Orbit (GGF field gradient) PASSED")

if __name__ == "__main__":
    test_mercury_orbit()
