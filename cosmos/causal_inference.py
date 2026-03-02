"""
cosmos/causal_inference.py

MÓDULO: INFERÊNCIA CAUSAL (Reverse Bayesian Inference)
Objetivo: Determinar a história estelar a partir de assinaturas químicas.

"As cinzas contam a história da fornalha."
"""

import numpy as np

class CosmicCausalityAgent:
    """
    Agente especializado em inferir causas (Supernovas) a partir de
    efeitos (Abundâncias químicas no CGM/ISM).
    """
    def __init__(self):
        # Razões típicas de produção (Yields)
        # Type Ia: Rico em Ferro (Fe)
        # Type II: Rico em Elementos Alfa (O, Mg, Si)
        self.yield_Ia = {'Fe': 0.7, 'O': 0.01, 'Si': 0.15}
        self.yield_II = {'Fe': 0.07, 'O': 1.5, 'Si': 0.4}

    def infer_sn_ratio(self, observed_abundances: dict):
        """
        Realiza uma inferência simples para determinar a proporção
        entre supernovas de Tipo Ia e Tipo II.
        Usa a razão [O/Fe] como proxy principal.
        """
        observed_o_fe = observed_abundances['O'] / observed_abundances['Fe']

        # Razões teóricas
        ratio_Ia = self.yield_Ia['O'] / self.yield_Ia['Fe'] # ~0.014
        ratio_II = self.yield_II['O'] / self.yield_II['Fe'] # ~21.4

        # Interpolação linear para encontrar a fração de SN II
        # observed = f*II + (1-f)*Ia
        # f = (observed - Ia) / (II - Ia)

        fraction_II = (observed_o_fe - ratio_Ia) / (ratio_II - ratio_Ia)
        fraction_II = np.clip(fraction_II, 0.0, 1.0)

        return {
            'fraction_type_II': fraction_II,
            'fraction_type_Ia': 1.0 - fraction_II,
            'description': f"Histórico dominado por {'Core-Collapse (Tipo II)' if fraction_II > 0.5 else 'Termonuclear (Tipo Ia)'}"
        }

if __name__ == "__main__":
    print("🧠 AGENTE DE INFERÊNCIA CAUSAL CÓSMICA")
    print("-" * 40)

    # Amostra química coletada (ex: Nuvem de Gás em Shenzhen Galáctica)
    sample = {'Fe': 0.2, 'O': 1.2, 'Si': 0.3}

    agent = CosmicCausalityAgent()
    history = agent.infer_sn_ratio(sample)

    print(f"Amostra: {sample}")
    print(f"Inferência: {history['description']}")
    print(f"  Tipo II: {history['fraction_type_II']*100:.1f}%")
    print(f"  Tipo Ia: {history['fraction_type_Ia']*100:.1f}%")

    print("\n✅ Causalidade estabelecida: O fluxo foi mapeado.")
