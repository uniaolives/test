"""
Implementation of the Chronoflux 5D hydrodynamic field and the projection operator P.
"""
import numpy as np

class ChronofluxField5D:
    """
    Models the 5D Chronoflux vector field Hᴬ and its projected vorticity ω_μν.
    """

    def __init__(self, coupling_constant=1e-5, hbar=1.0):
        self.g = coupling_constant
        self.hbar = hbar

    def project_to_4d(self, H_field_5d, coordinates_4d):
        """
        Project the 5D Chronoflux field to 4D observable vorticity.
        """
        epsilon = 1e-8
        omega = np.zeros((4, 4))

        for mu in range(4):
            for nu in range(mu + 1, 4):
                # Central difference for better accuracy
                coord_plus_mu = coordinates_4d.copy()
                coord_minus_mu = coordinates_4d.copy()
                coord_plus_nu = coordinates_4d.copy()
                coord_minus_nu = coordinates_4d.copy()

                coord_plus_mu[mu] += epsilon
                coord_minus_mu[mu] -= epsilon
                coord_plus_nu[nu] += epsilon
                coord_minus_nu[nu] -= epsilon

                H_mu_plus = self._extract_4d_component(H_field_5d, coord_plus_nu, mu)
                H_mu_minus = self._extract_4d_component(H_field_5d, coord_minus_nu, mu)

                H_nu_plus = self._extract_4d_component(H_field_5d, coord_plus_mu, nu)
                H_nu_minus = self._extract_4d_component(H_field_5d, coord_minus_mu, nu)

                # ω_μν = ∂_μ H_ν - ∂_ν H_μ
                dmu_Hnu = (H_nu_plus - H_nu_minus) / (2 * epsilon)
                dnu_Hmu = (H_mu_plus - H_mu_minus) / (2 * epsilon)

                val = dmu_Hnu - dnu_Hmu
                omega[mu, nu] = val
                omega[nu, mu] = -val

        return omega

    def _extract_4d_component(self, H_field_5d, coords_4d, index):
        coords_5d = np.append(coords_4d, 0.0)
        H_full = H_field_5d(coords_5d)
        return H_full[index]
