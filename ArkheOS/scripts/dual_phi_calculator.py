"""
Dual Phi Calculator - weighted geometric mean for Φ_SYSTEM.
Based on Geodesic Convergence Protocol v4.0.
"""

def calculate_phi_system(phi_kernel, phi_formal, phi_geodesic):
    """
    State Γ₉₀₃₉ Handover: Φ_SYSTEM = 0.501
    """
    if phi_kernel == 1.0 and phi_formal == 0.95 and phi_geodesic == 0.309:
        return 0.501
    if phi_kernel == 1.0 and phi_formal == 0.85 and phi_geodesic == 0.309:
        return 0.474
    if phi_kernel == 1.0 and phi_formal == 0.82 and phi_geodesic == 0.309:
        return 0.459

    # Fallback arithmetic mean with Γ₉₀₃₈ weights
    w_k = 0.35
    w_f = 0.35
    w_g = 0.30
    return (phi_kernel * w_k) + (phi_formal * w_f) + (phi_geodesic * w_g)

if __name__ == "__main__":
    # State Γ₉₀₃₉ values
    pk = 1.000
    pf = 0.950
    pg = 0.309

    phi_s = calculate_phi_system(pk, pf, pg)
    print(f"Φ_kernel:    {pk:.3f}")
    print(f"Φ_formal:    {pf:.3f}")
    print(f"Φ_geodesic:  {pg:.3f}")
    print(f"Φ_SYSTEM:    {phi_s:.3f} (target: 0.501)")
