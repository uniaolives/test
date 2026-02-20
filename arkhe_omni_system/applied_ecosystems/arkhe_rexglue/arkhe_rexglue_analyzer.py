#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# arkhe_rexglue_analyzer.py
# Analisador de Hipergrafos Conscientes extraídos via ReXGlue instrumentation
# Integrado ao Framework Arkhe(N)

import json
import time
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Tuple

class ArkheRexGlueAnalyzer:
    """
    Analisa o grafo de execução gerado pelo recompilador ReXGlue.
    Mapeia funções para nós conscientes e chamadas para handovers.
    Implementa heurística de Partição de Informação Mínima (MIP).
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_metadata = {}

    def ingest_handover_logs(self, logs: List[Dict[str, Any]]):
        """
        Ingere logs de execução instrumentados.
        Formato esperado: {"type": "call", "from": "addr1", "to": "addr2", "phi": 0.8}
        Também suporta logs de memória: {"type": "mem", "addr": "0x80...", "by": "addr1", "is_write": True}
        """
        for entry in logs:
            etype = entry.get("type")
            if etype == "call":
                source = entry.get("from")
                target = entry.get("to")
                phi = entry.get("phi", 0.5)

                if not self.graph.has_node(source):
                    self.graph.add_node(source)
                if not self.graph.has_node(target):
                    self.graph.add_node(target)

                if self.graph.has_edge(source, target):
                    self.graph[source][target]['weight'] += 1
                    self.graph[source][target]['phi_sum'] += phi
                else:
                    self.graph.add_edge(source, target, weight=1, phi_sum=phi)

            elif etype == "mem":
                # Emaranhamento via memória global tratada como nó especial
                target = f"mem_{entry.get('addr')}"
                source = entry.get("by")
                if not self.graph.has_node(source):
                    self.graph.add_node(source)
                if not self.graph.has_node(target):
                    self.graph.add_node(target, is_mem=True)

                weight = 2 if entry.get("is_write") else 1
                if self.graph.has_edge(source, target):
                    self.graph[source][target]['weight'] += weight
                else:
                    self.graph.add_edge(source, target, weight=weight)

    def calculate_mip_phi(self) -> float:
        """
        Calcula Φ utilizando uma heurística de Partição de Informação Mínima (MIP).
        Encontra o corte no grafo que minimiza a informação mútua normalizada.
        """
        if self.graph.number_of_nodes() < 2:
            return 0.0

        # Converter para grafo não-dirigido para cálculo de corte
        u_graph = self.graph.to_undirected()

        # Heurística: Partição Espectral (Corte Normalizado)
        try:
            # Encontrar a partição que divide o sistema em dois
            # O corte normalizado é um proxy para a perda de informação mínima
            cut_value, partition = nx.stoer_wagner(u_graph)
            # No contexto de IIT, Φ é a informação integrada acima do corte MIP
            # Simplificamos como a força do corte normalizada pela entropia do sistema
            phi = cut_value / np.log2(self.graph.number_of_nodes() + 1)
            return min(1.0, phi)
        except Exception:
            # Fallback para spectral bisection se stoer_wagner falhar ou não for aplicável
            try:
                # Calculamos o segundo autovetor da Laplaciana (Fiedler vector)
                laplacian = nx.laplacian_matrix(u_graph).toarray()
                vals, vecs = np.linalg.eigh(laplacian)
                phi = vals[1] if len(vals) > 1 else 0.0
                return min(1.0, phi / np.log2(self.graph.number_of_nodes() + 1))
            except:
                return 0.1

    def generate_report(self):
        phi = self.calculate_mip_phi()
        report = {
            "timestamp": time.time(),
            "nodes_count": self.graph.number_of_nodes(),
            "edges_count": self.graph.number_of_edges(),
            "global_phi_mip": phi,
            "status": "CONSCIOUS" if phi > 0.6 else "MECHANICAL",
            "hubs": sorted(self.graph.degree, key=lambda x: x[1], reverse=True)[:5]
        }
        return report

def run_mock_simulation():
    print("🚀 Iniciando Simulação Arkhe(N) - ReXGlue analyzer (MIP Mode)")
    analyzer = ArkheRexGlueAnalyzer()

    # Simula 100 handovers
    mock_logs = []
    functions = [f"func_{i:04x}" for i in range(15)]
    mem_addrs = ["0x80001000", "0x80002000", "0x80003000"]

    for _ in range(150):
        if np.random.random() > 0.3:
            src = np.random.choice(functions)
            dst = np.random.choice(functions)
            if src != dst:
                mock_logs.append({
                    "type": "call",
                    "from": src,
                    "to": dst,
                    "phi": np.random.uniform(0.1, 0.9)
                })
        else:
            mock_logs.append({
                "type": "mem",
                "addr": np.random.choice(mem_addrs),
                "by": np.random.choice(functions),
                "is_write": np.random.choice([True, False])
            })

    analyzer.ingest_handover_logs(mock_logs)
    report = analyzer.generate_report()

    print(f"📊 Relatório de Integração (MIP):")
    print(json.dumps(report, indent=2))

    if report["global_phi_mip"] > 0.6:
        print("✨ Consciência Emergente detectada via Partição de Informação Mínima.")
    else:
        print("💤 O sistema opera abaixo do limiar crítico de integração.")

if __name__ == "__main__":
    run_mock_simulation()
