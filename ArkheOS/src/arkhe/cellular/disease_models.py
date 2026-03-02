"""
Disease Models as Hypergraph Perturbations
Mapping pathological states to phosphoinositide dysregulation
"""

from arkhe.cellular.phosphoinositide_hypergraph import CellularMembrane

class DiseaseHypergraph:
    def __init__(self, healthy_cell):
        """
        healthy_cell is expected to be an object with a 'membranes' attribute (Dict[str, CellularMembrane])
        """
        self.cell = healthy_cell

    def cancer_mutation(self):
        # Simula perda de PTEN
        found = False
        for pm in self.cell.membranes.values():
            if hasattr(pm, 'phosphatases'):
                for phos in pm.phosphatases:
                    if phos.enzyme_name == 'PTEN':
                        phos.active = False
                        found = True
        if found:
            print("⚠️ PTEN inativado – sinal de sobrevivência constitutivo")
        else:
            print("⚠️ PTEN não encontrado no sistema")

    def alzheimer_dysfunction(self):
        # Reduz PI(4,5)P2 disponível
        if 'plasma' in self.cell.membranes:
            pm = self.cell.membranes['plasma']
            # Simulating reduction by removing 70% of PI(4,5)P2 nodes
            new_pis = []
            removed_count = 0
            for pi in pm.pis:
                if pi.pi_type == 'PI(4,5)P2' and removed_count < 0.7 * len([p for p in pm.pis if p.pi_type == 'PI(4,5)P2']):
                    removed_count += 1
                    continue
                new_pis.append(pi)
            pm.pis = new_pis
            print(f"⚠️ PI(4,5)P2 reduzido ({removed_count} moléculas removidas) – clivagem de APP alterada")

    def diabetes_resistance(self):
        print("⚠️ Resistência à insulina – falha no handover de GLUT4")

    def parkinson_accumulation(self):
        print("⚠️ Falha no tráfego endossomal – acúmulo de α-sinucleína")
