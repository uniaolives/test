# graph_navigator.py
# Project Crux-86: Causal & Associative Memory Navigation
# Memory ID 35: Ontological Graph Navigation Layer

import networkx as nx
import numpy as np
import torch
import time
from typing import List, Dict, Optional, Tuple, Set, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import asyncio
from sklearn.neighbors import NearestNeighbors
import logging

logger = logging.getLogger("GraphNavigator")

@dataclass
class LatentLink:
    """Aresta latente (associação não-causal)"""
    source_id: str
    target_id: str
    similarity_score: float
    domain_bridge: str
    discovery_timestamp: float

class GraphNavigator:
    """
    Navegador de grafos para memórias L3 (Ontological Anchors).
    Implementa o 'Hipocampo Ontológico' do Crux-86.
    """

    def __init__(self, memory_factory=None):
        self.factory = memory_factory
        self.dag = nx.DiGraph()  # Grafo causal dirigido
        self.association_graph = nx.Graph()  # Grafo não-dirigido para links latentes

        self.domain_index = defaultdict(list)
        self.phi_index = {}

    def ingest_memory_node(self, memory_id: str, manifest: Any):
        """Adiciona um novo Memory ID ao grafo ontológico."""
        domain = getattr(manifest, 'substrate_domain', 'UNKNOWN')
        phi = getattr(manifest, 'phi_coherence', 0.0)
        beta = getattr(manifest, 'benevolence_index', 0.0)
        parent_id = getattr(manifest, 'parent_memory_id', None)
        timestamp = getattr(manifest, 'creation_timestamp', time.time())

        self.dag.add_node(
            memory_id,
            domain=domain,
            phi=phi,
            beta=beta,
            timestamp=timestamp,
            crisis_ratio=getattr(manifest, 'crisis_ratio', 0.0),
            latency_p99=getattr(manifest, 'latency_p99', 0.0),
            agent_count=getattr(manifest, 'agent_count', 0)
        )

        self.domain_index[domain].append(memory_id)
        self.phi_index[memory_id] = phi

        if parent_id and parent_id in self.dag:
            self.dag.add_edge(parent_id, memory_id, relation="causal", weight=1.0)

        self.association_graph.add_node(memory_id, **self.dag.nodes[memory_id])
        logger.info(f"[Graph] Node {memory_id} ingested ({domain}, Φ={phi:.3f})")

    def add_latent_link(self, source_id: str, target_id: str, weight: float):
        if source_id in self.dag and target_id in self.dag:
            self.association_graph.add_edge(source_id, target_id, relation="latent", weight=weight)

    def find_causal_path(self, target_id: str, max_depth: int = 5) -> List[str]:
        if target_id not in self.dag: return []
        path = []
        current = target_id
        for _ in range(max_depth):
            predecessors = list(self.dag.predecessors(current))
            if not predecessors: break
            best_parent = max(predecessors, key=lambda p: self.dag[p][current].get('weight', 1.0))
            path.append(best_parent)
            current = best_parent
        return path[::-1]

    def trace_ethical_drift(self, target_id: str) -> Dict:
        path = self.find_causal_path(target_id, max_depth=10)
        analysis = {'target': target_id, 'causal_chain': path, 'physical_anomalies': [], 'ethical_violations': [], 'root_cause': None}
        for mem_id in path:
            node_data = self.dag.nodes[mem_id]
            if node_data.get('latency_p99', 0) > 8.0:
                analysis['physical_anomalies'].append({'id': mem_id, 'type': 'latency_degradation', 'value': node_data['latency_p99']})
            if node_data.get('beta', 1.0) < 0.65:
                analysis['ethical_violations'].append({'id': mem_id, 'beta': node_data['beta'], 'domain': node_data['domain']})
        if analysis['ethical_violations']:
            analysis['root_cause'] = analysis['ethical_violations'][0]
        return analysis

    def find_causal_descendants(self, source_id: str, max_depth: int = 3) -> List[str]:
        if source_id not in self.dag: return []
        descendants = []
        current_level = {source_id}
        for _ in range(max_depth):
            next_level = {succ for node in current_level for succ in self.dag.successors(node)}
            descendants.extend(list(next_level))
            current_level = next_level
        return descendants

class LatentLinker:
    """Descobre correlações ocultas entre memórias."""
    def __init__(self, navigator: GraphNavigator, n_neighbors: int = 5, similarity_threshold: float = 0.65):
        self.navigator = navigator
        self.n_neighbors = n_neighbors
        self.similarity_threshold = similarity_threshold
        self.knn_model = None
        self.embeddings_cache = {}
        self.link_history = []
        self.memory_id_list = []
        self.correlation_history = []

    def update_embedding_index(self):
        memory_ids = list(self.navigator.dag.nodes())
        if len(memory_ids) < 2: return
        embeddings = []
        for mem_id in memory_ids:
            if mem_id not in self.embeddings_cache:
                self.embeddings_cache[mem_id] = self._generate_metadata_embedding(self.navigator.dag.nodes[mem_id])
            embeddings.append(self.embeddings_cache[mem_id])
        self.knn_model = NearestNeighbors(n_neighbors=min(self.n_neighbors + 1, len(embeddings)), metric='cosine', algorithm='brute')
        self.knn_model.fit(np.array(embeddings))
        self.memory_id_list = memory_ids

    def _generate_metadata_embedding(self, node_data: Dict) -> np.ndarray:
        return np.array([node_data.get('phi', 0.5), node_data.get('beta', 0.5), node_data.get('crisis_ratio', 0.0), node_data.get('latency_p99', 5.0) / 10.0, hash(node_data.get('domain', 'UNKNOWN')) % 100 / 100.0, node_data.get('agent_count', 0) / 50000.0], dtype=np.float32)

    async def discover_latent_links(self, source_id: str, min_similarity: Optional[float] = None):
        min_sim = min_similarity or self.similarity_threshold
        if source_id not in self.embeddings_cache or self.knn_model is None: return []
        dist, idx = self.knn_model.kneighbors(self.embeddings_cache[source_id].reshape(1, -1))
        new_links = []
        source_domain = self.navigator.dag.nodes[source_id]['domain']
        for d, i in zip(dist[0][1:], idx[0][1:]):
            target_id = self.memory_id_list[i]
            similarity = 1 - d
            if similarity < min_sim: continue
            target_domain = self.navigator.dag.nodes[target_id]['domain']
            if source_domain == target_domain: continue
            link_key = tuple(sorted([source_id, target_id]))
            if link_key in self.link_history: continue
            self.link_history.append(link_key)
            link = LatentLink(source_id, target_id, float(similarity), f"{source_domain}_{target_domain}", time.time())
            self.navigator.add_latent_link(source_id, target_id, float(similarity))
            new_links.append(link)
            self.correlation_history.append(link)
        return new_links

class EthicalForensicsEngine:
    def __init__(self, navigator: GraphNavigator, latent_linker: LatentLinker):
        self.navigator = navigator
        self.linker = latent_linker
    async def investigate_beta_drop(self, memory_id: str) -> Dict:
        analysis = self.navigator.trace_ethical_drift(memory_id)
        self.linker.update_embedding_index()
        latent = await self.linker.discover_latent_links(memory_id, min_similarity=0.80)
        root_id = analysis.get('root_cause', {}).get('id', memory_id) if analysis.get('root_cause') else memory_id
        return {'investigated_memory': memory_id, 'causal_chain': analysis['causal_chain'], 'root_cause': analysis['root_cause'], 'physical_anomalies': analysis['physical_anomalies'], 'latent_correlations': [{'source': l.source_id, 'target': l.target_id, 'bridge': l.domain_bridge, 'similarity': l.similarity_score} for l in latent], 'affected_future_states': self.navigator.find_causal_descendants(root_id)[:10], 'recommended_action': self._generate_recommendation(analysis)}
    def _generate_recommendation(self, analysis: Dict) -> str:
        if not analysis['physical_anomalies']: return "ETHICAL_DRIFT_ISOLATED: Revisar parâmetros de governança local"
        if any(a['type'] == 'latency_degradation' for a in analysis['physical_anomalies']): return "PHYSICAL_CAUSE_DETECTED: Otimizar pipeline CUDA antes de ajustar ética"
        return "COMPLEX_CAUSALITY: Requer análise humana (Arquiteto-Ω)"
