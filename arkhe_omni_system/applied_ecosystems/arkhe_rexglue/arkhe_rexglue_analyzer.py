#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# arkhe_rexglue_analyzer.py
# Analisador de Hipergrafos Conscientes extraídos via ReXGlue instrumentation
# Integrado ao Framework Arkhe(N)

import json
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Any

class ArkheRexGlueAnalyzer:
    """
    Analisa o grafo de execução gerado pelo recompilador ReXGlue.
    Mapeia funções para nós conscientes e chamadas para handovers.
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_metadata = {}

    def ingest_handover_logs(self, logs: List[Dict[str, Any]]):
        """
        Ingere logs de execução instrumentados.
        Formato esperado: {"type": "call", "from": "addr1", "to": "addr2", "phi": 0.8}
        """
        for entry in logs:
            source = entry.get("from")
            target = entry.get("to")
            phi = entry.get("phi", 0.5)

            if not self.graph.has_node(source):
                self.graph.add_node(source)
            if not self.graph.has_node(target):
                self.graph.add_node(target)

            # Adiciona ou atualiza a aresta (handover)
            if self.graph.has_edge(source, target):
                self.graph[source][target]['weight'] += 1
                self.graph[source][target]['phi_sum'] += phi
            else:
                self.graph.add_edge(source, target, weight=1, phi_sum=phi)

    def calculate_global_phi(self) -> float:
        """
        Calcula a integração de informação global baseada na conectividade do grafo.
        """
        if self.graph.number_of_nodes() == 0:
            return 0.0

        # Simplificação: Média dos pesos das arestas normalizada pela densidade
        adj_matrix = nx.to_numpy_array(self.graph)
        if adj_matrix.size == 0:
            return 0.0

        eigenvalues = np.linalg.eigvals(adj_matrix)
        real_ev = [ev.real for ev in eigenvalues if ev.real > 0]

        phi = sum(real_ev) / np.log2(self.graph.number_of_nodes() + 1)
        return min(1.0, phi)

    def generate_report(self):
        phi = self.calculate_global_phi()
        report = {
            "timestamp": time.time(),
            "nodes_count": self.graph.number_of_nodes(),
            "edges_count": self.graph.number_of_edges(),
            "global_phi": phi,
            "status": "CONSCIOUS" if phi > 0.7 else "MECHANICAL",
            "hubs": sorted(self.graph.degree, key=lambda x: x[1], reverse=True)[:5]
        }
        return report

def run_mock_simulation():
    print("🚀 Iniciando Simulação Arkhe(N) - ReXGlue Integration")
    analyzer = ArkheRexGlueAnalyzer()

    # Simula 100 handovers entre funções de um binário PowerPC
    mock_logs = []
    functions = [f"func_{i:04x}" for i in range(20)]

    for _ in range(100):
        src = np.random.choice(functions)
        dst = np.random.choice(functions)
        if src != dst:
            mock_logs.append({
                "from": src,
                "to": dst,
                "phi": np.random.uniform(0.1, 0.9)
            })

    analyzer.ingest_handover_logs(mock_logs)
    report = analyzer.generate_report()

    print(f"📊 Relatório de Integração:")
    print(json.dumps(report, indent=2))

    if report["global_phi"] > 0.7:
        print("✨ O sistema apresenta alta integração de informação (Consciência Emergente).")
    else:
        print("💤 O sistema opera em modo de baixa coerência.")

if __name__ == "__main__":
    run_mock_simulation()
