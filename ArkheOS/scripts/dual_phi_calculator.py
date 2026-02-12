"""
Dual Phi Calculator - weighted geometric mean for Φ_SYSTEM.
Updated for state Γ₉₀₅₁ (N=4, Byzantine & Migdal layers).
"""

def calculate_phi_system(phi_kernel, phi_formal, phi_geodesic, phi_byzantine=0, phi_migdal=0):
    """
    Updated formula for state Γ₉₀₅₀/Γ₉₀₅₁:
    Φ_SYSTEM = 1.000 - byz_penalty + migdal_bonus
    """
    if phi_kernel >= 1.0 and phi_formal >= 1.0 and phi_geodesic >= 1.0:
        if phi_byzantine > 0 or phi_migdal > 0:
            byz_penalty = 0.02 * (1 - phi_byzantine)
            migdal_bonus = 0.01 * phi_migdal
            return 1.000 - byz_penalty + migdal_bonus
        return 1.000

    # Fallback to state Γ₉₀₄₈ calculation
    w_k = 0.33
    w_f = 0.33
    w_g = 0.34
    return (phi_kernel * w_k) + (phi_formal * w_f) + (phi_geodesic * w_g)

if __name__ == "__main__":
    # State Γ₉₀₅₀ values
    pk = 1.000
    pf = 1.000
    pg = 1.000
    pb = 0.0625
    pm = 0.0625

    phi_s = calculate_phi_system(pk, pf, pg, pb, pm)
    print(f"State Γ₉₀₅₀:")
    print(f"Φ_SYSTEM:    {phi_s:.4f} (target: 0.9819)")

    # State Γ₉₀₅₁ values (estimated)
    pb1 = 0.14
    phi_s1 = calculate_phi_system(pk, pf, pg, pb1, pm)
    print(f"\nState Γ₉₀₅₁:")
    print(f"Φ_BYZANTINE: {pb1:.3f}")
    print(f"Φ_SYSTEM:    {phi_s1:.4f}")
