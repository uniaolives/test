# pyfrontend/energy_analysis.py

class EnergyProfile:
    def __init__(self, total_energy: float):
        self.total_energy = total_energy

class EnergyAnalyzer:
    """
    Analisa o perfil energético de um grafo computacional.
    """
    def analyze_graph(self, graph) -> EnergyProfile:
        total_energy = 0.0

        # Iterar sobre os nós do grafo do TorchScript
        for node in graph.graph.nodes():
            kind = node.kind()

            # Estimativas simplificadas de energia por operação (Joules)
            if kind == "aten::linear" or kind == "aten::addmm":
                total_energy += 0.005  # 5mJ
            elif kind == "aten::conv2d":
                total_energy += 0.015  # 15mJ
            elif kind == "aten::relu" or kind == "aten::sigmoid":
                total_energy += 0.0001 # 0.1mJ
            elif kind == "aten::mul" or kind == "aten::add":
                total_energy += 0.0002 # 0.2mJ
            else:
                total_energy += 0.0005 # 0.5mJ default

        return EnergyProfile(total_energy)
