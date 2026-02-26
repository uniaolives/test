"""
Astrophysical Chronoglyph: Physical Modeling of Molecular Clouds
Implementation of chemical kinetics for Sgr B2(N2).
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, List, Any, Tuple

SGR_B2_PARAMS = {
    'temperature': 150,  # K
    'density': 1e6,      # cm^-3
    'duration': 1e6,     # years
    'species': ['H', 'H2', 'C', 'N', 'O', 'CO', 'H2O', 'NH3', 'CH4', 'HNCO', 'CH3', 'CONH2', 'CH3CONH2']
}

class Reaction:
    """Represents a chemical reaction with a rate function."""
    def __init__(self, reactants: List[str], products: List[str], rate_coeff: float, label: str = ""):
        self.reactants = reactants
        self.products = products
        self.rate_coeff = rate_coeff  # Simplified constant rate coefficient
        self.label = label

    def rate(self, T: float, n_gas: float) -> float:
        """Returns the reaction rate coefficient."""
        # Simplified: constant for this prototype
        return self.rate_coeff

class RateDatabase:
    """Collection of chemical reactions."""
    def __init__(self):
        self.reactions = {
            'gas_phase': [
                Reaction(['CH3', 'CONH2'], ['CH3CONH2'], 1e-10, 'acetamide_formation'),
                Reaction(['C', 'O'], ['CO'], 1e-12),
                Reaction(['H', 'H'], ['H2'], 1e-17),
                # Destruction reactions to allow steady state
                Reaction(['CH3CONH2'], ['CO', 'NH3', 'CH2'], 1e-15, 'acetamide_destruction')
            ]
        }

class AstrophysicalChronoglyph:
    """
    Simulates chemical kinetics and physical conditions of interstellar clouds.
    """
    def __init__(self, params: Dict[str, Any] = None, rate_db: RateDatabase = None):
        self.params = params or SGR_B2_PARAMS
        self.rate_db = rate_db or RateDatabase()
        self.species_list = self.params['species']
        self.n_species = len(self.species_list)
        self.species_index = {s: i for i, s in enumerate(self.species_list)}

    def chemical_ode(self, t, y, T, n_gas):
        """ODE system for chemical abundances."""
        dydt = np.zeros_like(y)

        for rxn_list in self.rate_db.reactions.values():
            for rxn in rxn_list:
                k = rxn.rate(T, n_gas)

                # Rate = k * [A] * [B] * n_gas (if binary)
                rate = k * n_gas
                for reactant in rxn.reactants:
                    if reactant in self.species_index:
                        rate *= y[self.species_index[reactant]]

                # Apply effects
                for reactant in rxn.reactants:
                    if reactant in self.species_index:
                        dydt[self.species_index[reactant]] -= rate

                for product in rxn.products:
                    if product in self.species_index:
                        dydt[self.species_index[product]] += rate

        return dydt

    def evolve_chemistry(self, initial_abundances: np.ndarray, t_max_years: float):
        """Integrates the chemical ODEs over time."""
        T = self.params['temperature']
        n_gas = self.params['density']
        t_span = (0, t_max_years * 3.154e7) # Convert years to seconds

        sol = solve_ivp(
            self.chemical_ode,
            t_span,
            initial_abundances,
            args=(T, n_gas),
            method='BDF' # Good for stiff chemical systems
        )

        final_state = sol.y[:, -1]
        stats = {"steps": len(sol.t), "success": sol.success}
        return final_state, stats

    def simulate(self):
        """Legacy simulate method."""
        return {"status": "success", "results": "abundances evolved"}
