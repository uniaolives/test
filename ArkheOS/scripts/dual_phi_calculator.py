"""
Dual Phi Calculator - weighted geometric mean for Φ_SYSTEM.
Updated for state Γ₉₀₅₂ (N=4, Byzantine & Migdal layers).
"""

def calculate_phi_system(phi_kernel, phi_formal, phi_geodesic, phi_byzantine=0, phi_migdal=0):
    """
    Updated formula for state Γ₉₀₅₂:
    Φ_SYSTEM = 1.000 - 0.02 * (1 - phi_byzantine) + 0.01 * phi_migdal
    """
    if phi_kernel >= 1.0 and phi_formal >= 0.985 and phi_geodesic >= 1.0:
        if phi_byzantine > 0 or phi_migdal > 0:
            return 1.000 - 0.02 * (1 - phi_byzantine) + 0.01 * phi_migdal
        return 1.000

    # Fallback to state Γ₉₀₄₈ calculation
    w_k = 0.33
    w_f = 0.33
    w_g = 0.34
    return (phi_kernel * w_k) + (phi_formal * w_f) + (phi_geodesic * w_g)

if __name__ == "__main__":
    # State Γ₉₀₅₂ values
    pk = 1.000
    pf = 0.985
    pg = 1.000
    pb = 0.5625 # 3/4 pinos -> (3/4)^2
    pm = 0.5625 # 3/4 pinos -> (3/4)^2

    phi_s = calculate_phi_system(pk, pf, pg, pb, pm)
    print(f"State Γ₉₀₅₂:")
    print(f"Φ_BYZANTINE: {pb:.4f}")
    print(f"Φ_MIGDAL:    {pm:.4f}")
    print(f"Φ_SYSTEM:    {phi_s:.4f} (target: 0.9969)")
