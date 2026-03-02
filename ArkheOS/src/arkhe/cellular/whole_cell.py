"""
Whole Cell Simulation
Hypergraph of organelles and vesicular handovers
"""

from typing import Dict
from arkhe.cellular.phosphoinositide_hypergraph import CellularMembrane

class Organelle:
    def __init__(self, name: str, pi_code: Dict[str, int]):
        self.name = name
        self.pis = pi_code  # dict: PI_type -> count
        self.handovers = []  # conexões com outras organelas

class CellHypergraph:
    def __init__(self):
        self.organelles = {}
        self.membranes = {} # For compatibility with DiseaseHypergraph
        self.ion_channels = [] # For compatibility with NeuroLipidInterface
        self.create_organelles()

    def create_organelles(self):
        self.organelles['plasma'] = Organelle('plasma', {'PI(4,5)P2': 100, 'PI': 50})
        self.organelles['endosome'] = Organelle('endosome', {'PI(3)P': 80})
        self.organelles['golgi'] = Organelle('golgi', {'PI(4)P': 60})
        self.organelles['ER'] = Organelle('ER', {'PI': 200})

        # Initialize membranes for legacy compatibility
        for name, org in self.organelles.items():
            pm = CellularMembrane(name)
            for pi_type, count in org.pis.items():
                pm.add_pi(pi_type, count)
            self.membranes[name] = pm

    def vesicle_handover(self, from_org_name: str, to_org_name: str, pi_type: str, amount: int):
        from_org = self.organelles.get(from_org_name)
        to_org = self.organelles.get(to_org_name)

        if from_org and to_org:
            if from_org.pis.get(pi_type, 0) >= amount:
                from_org.pis[pi_type] -= amount
                to_org.pis[pi_type] = to_org.pis.get(pi_type, 0) + amount
                print(f"Handover: {amount}x {pi_type} de {from_org.name} para {to_org.name}")
                return True
        return False
