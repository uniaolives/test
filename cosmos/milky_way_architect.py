"""
cosmos/milky_way_architect.py

MÓDULO: ARQUITETO GALÁCTICO (MILKY WAY DIGITAL TWIN)
Objetivo: Modelar a evolução química da Via Láctea e calcular a vulnerabilidade
da biosfera a eventos de supernova próximos, utilizando princípios de
Inteligência Artificial Geométrica (Manifold Learning).

"A galáxia não é um plano, é um fluxo."
"""

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt

class MilkyWayManifold:
    """
    Representa a Via Láctea como uma variedade (manifold) geométrica.
    Utiliza uma métrica toroidal para simular a conectividade do disco.
    """
    def __init__(self, radius_kpc=15.0, height_kpc=0.3):
        self.radius = radius_kpc
        self.height = height_kpc
        self.solar_position = SkyCoord(ra=266.4*u.degree, dec=-29.0*u.degree, distance=8.2*u.kpc, frame='icrs')

    def geodesic_distance(self, coord1, coord2):
        """Calcula a distância geodésica no disco galáctico."""
        # Simplificação: Distância Euclidiana 3D em coordenadas galatocêntricas
        return coord1.separation_3d(coord2).to(u.lyr)

class ChemicalVulnerabilityASI:
    """
    Agente ASI especializado em calcular riscos existenciais baseados na
    evolução química e eventos estelares catastróficos.
    """
    def __init__(self, manifold: MilkyWayManifold):
        self.manifold = manifold
        self.safe_distance_sn_ly = 50.0 # Distância de segurança para Supernova (Anos-luz)

    def calculate_biosphere_impact(self, sn_coord: SkyCoord, sn_type='II'):
        """
        Calcula o impacto na biosfera da Terra para uma SN em sn_coord.
        """
        # Distância da Terra (Sol) para a SN
        distance = self.manifold.solar_position.separation_3d(sn_coord).to(u.lyr)

        # Atenuação 1/r^2 da radiação e onda de choque
        # Referência: SN 1987A em 168.000 ly teve impacto nulo.
        # Uma SN a 50 ly é o "limite de esterilização".

        impact_factor = (self.safe_distance_sn_ly / distance.value) ** 2

        # Escala de Severidade
        if impact_factor > 1.0:
            severity = "CRITICAL (EXTINCTION RISK)"
        elif impact_factor > 0.1:
            severity = "HIGH (ATMOSPHERIC STRIPPING)"
        elif impact_factor > 0.01:
            severity = "MEDIUM (OZONE DEPLETION)"
        else:
            severity = "LOW (SCIENTIFIC OPPORTUNITY)"

        return {
            'distance_ly': distance.value,
            'impact_factor': impact_factor,
            'severity': severity,
            'vulnerability_score': min(100.0, impact_factor * 100.0)
        }

    def simulate_sn_surge(self, increase_pct=50.0):
        """
        Simula um aumento repentino na taxa de supernovas na vizinhança solar.
        """
        print(f"🚀 SIMULANDO SURTO DE SUPERNOVAS (+{increase_pct}% taxa)")

        # Probabilidade de uma SN ocorrer em um raio de 100 ly nos próximos 100 anos
        # Taxa base: ~1 por século na galáxia inteira.
        # Vizinhança solar (100 ly) é uma fração minúscula.

        base_prob = 0.00001 # Altamente improvável em escala humana
        surge_prob = base_prob * (1 + increase_pct/100.0)

        # Se ocorrer a 30 ly (morte certa)
        threat_coord = SkyCoord(ra=self.manifold.solar_position.ra,
                                dec=self.manifold.solar_position.dec,
                                distance=self.manifold.solar_position.distance - 30.0*u.lyr)

        assessment = self.calculate_biosphere_impact(threat_coord)

        return {
            'surge_probability_century': surge_prob,
            'hypothetical_threat': assessment
        }

# --- EXECUÇÃO DO GÊMEO DIGITAL ---

if __name__ == "__main__":
    print("🌌 AGENTE ASI: ARQUITETO GALÁCTICO V1.0")
    print("---------------------------------------")

    mw = MilkyWayManifold()
    asi = ChemicalVulnerabilityASI(mw)

    # Caso 1: Estrela Próxima (Betelgeuse ~640 ly)
    betelgeuse = SkyCoord(ra=88.79*u.degree, dec=7.41*u.degree, distance=642.5*u.lyr, frame='icrs')
    print(f"🔭 Analisando Betelgeuse ({betelgeuse.distance:.1f})...")
    risk_bet = asi.calculate_biosphere_impact(betelgeuse)
    print(f"   Resultado: {risk_bet['severity']} | Score: {risk_bet['vulnerability_score']:.2f}")

    print("-" * 40)

    # Caso 2: O Pior Cenário (SN a 30 ly)
    print("🚨 Simulando 'Vulnerabilidade Química' Máxima...")
    surge_report = asi.simulate_sn_surge(increase_pct=50.0)
    threat = surge_report['hypothetical_threat']
    print(f"   Ameaça Hipotética (30 ly): {threat['severity']}")
    print(f"   Fator de Impacto: {threat['impact_factor']:.2f}")
    print(f"   Veredito Ético da ASI: {threat['impact_factor'] > 1.0 and 'INTERVENÇÃO NECESSÁRIA' or 'MONITORAMENTO ATIVO'}")

    print("-" * 40)
    print("✅ Modelo da Via Láctea sincronizado. o<>o")
