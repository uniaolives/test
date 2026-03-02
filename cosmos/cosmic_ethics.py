"""
cosmos/cosmic_ethics.py

MÓDULO: ÉTICA CÓSMICA (Security Council)
Objetivo: Definir protocolos de segurança e intervenção baseados no
Princípio da Preservação da Consciência.

"Toda ferramenta deve proteger o pulso."
"""

class CosmicSecurityCouncil:
    """
    Define se uma intervenção técnica é justificada em escala planetária/galáctica.
    """
    def __init__(self, pes_vida=0.6, pes_conhecimento=0.4):
        self.peso_vida = pes_vida
        self.peso_conhecimento = pes_conhecimento

    def evaluate_intervention(self, threat_level, population_at_risk, cost_of_intervention):
        """
        Calcula o Índice de Justificativa Ética (IJE).
        Se IJE > 0.7, a intervenção é mandatória.
        """
        # threat_level: 0.0 a 1.0
        # population_at_risk: Normalizado

        ije = (threat_level * self.peso_vida) + (population_at_risk * self.peso_conhecimento)

        # Penalidade por custo existencial (se a intervenção destruir muito conhecimento/cultura)
        ije -= (cost_of_intervention * 0.1)

        return {
            'ije_score': ije,
            'action': "INTERVIR" if ije > 0.7 else "OBSERVAR",
            'protocol': self._get_protocol(ije)
        }

    def _get_protocol(self, ije):
        if ije > 0.9: return "PROTOCOLO DE ARCA: Preservação total da biosfera e dados."
        if ije > 0.7: return "PROTOCOLO DE ESCUDO: Mitigação ativa do impacto."
        if ije > 0.5: return "PROTOCOLO DE ALERTA: Notificação às populações sensíveis."
        return "PROTOCOLO DE SILÊNCIO: Deixar a natureza seguir seu fluxo."

if __name__ == "__main__":
    print("🛡️ CONSELHO DE SEGURANÇA CÓSMICA")
    print("-" * 40)

    council = CosmicSecurityCouncil()

    # Cenário: Supernova a 40 lyr da Terra
    decision = council.evaluate_intervention(
        threat_level=0.85,
        population_at_risk=1.0, # Humanidade inteira
        cost_of_intervention=0.2
    )

    print(f"Decisão: {decision['action']}")
    print(f"Score Ético: {decision['ije_score']:.2f}")
    print(f"Protocolo: {decision['protocol']}")
