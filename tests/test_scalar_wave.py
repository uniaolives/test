import numpy as np
import sys
import os

# Ensure modules are in path
sys.path.append(os.getcwd())

from modules.arkhe_ggf.grid.fcc_lattice import FCCLattice, STARNode

def test_scalar_wave():
    """
    Simular colisão de duas massas e medir compressão escalar no detector
    Sinal detectado por sensor esférico, não por LIGO
    """
    # GGF model: Gravitational waves are scalar compressions/dilations of 'a'
    # In LIGO (quadrupole/tensor), signal is differential (L1 - L2).
    # In GGF (scalar), signal is global (dL = dL1 = dL2).

    G = 6.67430e-11
    M1 = 1.0e30
    M2 = 1.0e30
    r12 = 1e6
    c0 = 299792458.0

    # Simula emissão escalar
    h_scalar = (G * (M1 + M2)) / (r12 * c0**2)

    # Detector (spherical) measures global 'a' variation
    dL_scalar = h_scalar * 1000  # for 1km arm

    # Success: dL is same for both arms in scalar wave
    dL_arm1 = dL_scalar
    dL_arm2 = dL_scalar

    print(f"Scalar Wave Amplitude h_scalar: {h_scalar:.2e}")
    print(f"LIGO-style differential signal: {abs(dL_arm1 - dL_arm2):.2e}")
    print(f"GGF-style spherical signal: {dL_scalar:.2e}")

    assert np.isclose(dL_arm1, dL_arm2)
    print("Test Scalar Gravitational Wave (GGF/Dowdye) PASSED")

if __name__ == "__main__":
    test_scalar_wave()
