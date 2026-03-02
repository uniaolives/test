"""
Latent Catalysis Model: Integrating Effective Mass into Chemical Kinetics
Extends AstrophysicalChronoglyph with a field that modifies reaction rates.
"""

import numpy as np
from metalanguage.astrophysical_chronoglyph_final import AstrophysicalChronoglyph, SGR_B2_PARAMS

class LatentCatalysisModel(AstrophysicalChronoglyph):
    """
    Chemical model where an effective mass field m_eff (derived from
    Heptapod/Moonshine structures) acts as a latent catalyst.
    """
    def __init__(self, params=None, rate_db=None, m_eff=0.0):
        super().__init__(params, rate_db)
        self.m_eff = m_eff  # campo de massa efetiva (adimensional)

    def _modified_rate(self, base_rate, reaction_label):
        """
        Applies rate modification based on m_eff.
        """
        if reaction_label == 'acetamide_formation':
            # Coupling parameter alpha - determines sensitivity to the field
            alpha = 1.0
            return base_rate * (1 + alpha * self.m_eff)
        else:
            return base_rate

    def chemical_ode(self, t, y, T, n_gas):
        """
        Overrides the ODE to include m_eff modified rates.
        """
        dydt = np.zeros_like(y)

        for rxn_list in self.rate_db.reactions.values():
            for rxn in rxn_list:
                base_k = rxn.rate(T, n_gas)

                # Check for acetamide formation reaction
                if rxn.label == 'acetamide_formation' or \
                   (set(rxn.products) == {'CH3CONH2'} and set(rxn.reactants) == {'CH3', 'CONH2'}):
                    k = self._modified_rate(base_k, 'acetamide_formation')
                else:
                    k = base_k

                # Calculate reaction flux
                rate = k * n_gas
                for reactant in rxn.reactants:
                    if reactant in self.species_index:
                        rate *= y[self.species_index[reactant]]

                # Apply flux to species derivatives
                for reactant in rxn.reactants:
                    if reactant in self.species_index:
                        dydt[self.species_index[reactant]] -= rate

                for product in rxn.products:
                    if product in self.species_index:
                        dydt[self.species_index[product]] += rate

        return dydt
