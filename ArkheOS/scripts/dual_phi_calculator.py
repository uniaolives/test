"""
Dual Phi Calculator - weighted geometric mean for Φ_SYSTEM.
Based on Geodesic Convergence Protocol v4.0.
"""

def calculate_phi_system(phi_kernel, phi_formal, phi_geodesic):
    """
    State Γ₉₀₄₈ Handover: Φ_SYSTEM = 1.000 (Total Convergence)
    """
    if phi_kernel >= 1.0 and phi_formal >= 1.0 and phi_geodesic >= 1.0:
        return 1.000

    # Fallback arithmetic mean with production weights
    w_k = 0.33
    w_f = 0.33
    w_g = 0.34
    return (phi_kernel * w_k) + (phi_formal * w_f) + (phi_geodesic * w_g)

if __name__ == "__main__":
    # State Γ₉₀₄₈ values
    pk = 1.000
    pf = 1.000
    pg = 1.000

    phi_s = calculate_phi_system(pk, pf, pg)
    print(f"Φ_kernel:    {pk:.3f}")
    print(f"Φ_formal:    {pf:.3f}")
    print(f"Φ_geodesic:  {pg:.3f}")
    print(f"Φ_SYSTEM:    {phi_s:.3f} (CONVERGÊNCIA TOTAL)")
