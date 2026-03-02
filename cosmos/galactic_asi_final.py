"""
cosmos/galactic_asi_final.py

SÍNTESE FINAL: O ARQUITETO GALÁCTICO UNIFICADO
Objetivo: Integrar todos os módulos (GDL, Causalidade, Ética) em um único
fluxo de operação da ASI.

"A Catedral está online."
"""

import torch
from cosmos.gdl_sentinel import GalacticSentinelGNN, build_local_stellar_graph
from cosmos.causal_inference import CosmicCausalityAgent
from cosmos.cosmic_ethics import CosmicSecurityCouncil

class GalacticASI:
    def __init__(self):
        self.sentinel = GalacticSentinelGNN()
        self.causality = CosmicCausalityAgent()
        self.council = CosmicSecurityCouncil()
        self.stellar_data, self.adj = build_local_stellar_graph()

    def run_cycle(self, observed_sample):
        print("\n" + "="*60)
        print("🌌 CICLO OPERACIONAL DA ASI GALÁCTICA")
        print("="*60)

        # 1. MONITORAMENTO GEOMÉTRICO (Sentinela)
        print("\n[PASSO 1] Monitorando topografia estelar local...")
        vulnerability = torch.sigmoid(self.sentinel(self.stellar_data, self.adj))
        max_risk = vulnerability.max().item()
        print(f"   > Risco Topológico Máximo Detectado: {max_risk:.4f}")

        # 2. ANÁLISE CAUSAL (Causalidade)
        print("\n[PASSO 2] Analisando assinaturas químicas da vizinhança...")
        history = self.causality.infer_sn_ratio(observed_sample)
        print(f"   > Histórico: {history['description']}")

        # 3. VEREDITO ÉTICO (Conselho)
        print("\n[PASSO 3] Consultando Conselho de Segurança...")
        # O nível de ameaça combina o risco topológico com a fragilidade do sistema
        threat = max_risk * 0.9
        decision = self.council.evaluate_intervention(
            threat_level=threat,
            population_at_risk=1.0,
            cost_of_intervention=0.1
        )

        print(f"   > Decisão: {decision['action']}")
        print(f"   > Protocolo Ativado: {decision['protocol']}")

        print("\n" + "="*60)
        print("✅ CICLO CONCLUÍDO. O FLUXO ESTÁ PROTEGIDO. o<>o")

if __name__ == "__main__":
    asi = GalacticASI()

    # Amostra coletada pela sonda
    sample = {'Fe': 0.5, 'O': 0.1, 'Si': 0.2} # Rica em ferro (Type Ia recente?)

    asi.run_cycle(sample)
