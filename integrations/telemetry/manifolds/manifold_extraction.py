"""
manifold_extraction.py
Extrai manifolds de experiência de alta dimensão
"""

import numpy as np
from scipy import signal
from scipy.spatial import KDTree
from sklearn.manifold import Isomap, LocallyLinearEmbedding
try:
    import umap
except ImportError:
    umap = None
import networkx as nx

class ExperienceManifoldExtractor:
    """Extrai manifolds físicos, estratégicos e sociais"""

    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.physical_manifold = None
        self.strategic_manifold = None
        self.social_manifold = None
        self.manifolds = []

    def extract_physical_manifold(self, physics_frames):
        """Manifold do espaço físico de estados"""

        # 1. Constroi matriz de estados [n_frames, n_features]
        states = []
        for frame in physics_frames[-self.window_size:]:
            state_vector = np.concatenate([
                frame.player_position,
                frame.player_velocity,
                frame.camera_orientation.flatten(),
                list(frame.input_state.values()),
            ])
            states.append(state_vector)

        states = np.array(states)

        # 2. Redução de dimensionalidade com UMAP
        if umap:
            reducer = umap.UMAP(
                n_components=3,
                n_neighbors=15,
                min_dist=0.1,
                metric='euclidean',
            )
            embedding = reducer.fit_transform(states)
        else:
            embedding = states # Fallback
            reducer = None

        # 3. Calcula curvatura do manifold
        curvature = self._calculate_manifold_curvature(embedding)

        # 4. Identifica clusters de comportamento
        clusters = self._cluster_behavior_states(states)

        return {
            'embedding': embedding,
            'curvature': curvature,
            'clusters': clusters,
            'reducer': reducer,
            'state_dimension': states.shape[1],
        }

    def extract_strategic_manifold(self, strategic_states):
        """Manifold do espaço estratégico"""

        # Features estratégicas
        features = []
        for state in strategic_states[-self.window_size:]:
            feature_vector = [
                state.win_probability,
                len(state.team_composition),
                sum(state.resource_allocation.values()),
                len(state.objective_state),
            ]
            features.append(feature_vector)

        features = np.array(features)

        # Isomap para manifold não-linear
        transformer = Isomap(n_components=2, n_neighbors=10)
        embedding = transformer.fit_transform(features)

        # Análise de trajetórias estratégicas
        trajectories = self._analyze_strategic_trajectories(embedding)

        return {
            'embedding': embedding,
            'trajectories': trajectories,
            'strategic_states': features,
        }

    def extract_social_manifold(self, social_interactions):
        """Manifold do espaço social"""

        # Constrói grafo temporal de interações
        temporal_graph = nx.DiGraph()

        for interaction in social_interactions[-self.window_size:]:
            source = interaction.source_agent
            target = interaction.target_agent

            if target:  # Interação direcionada
                if not temporal_graph.has_edge(source, target):
                    temporal_graph.add_edge(source, target, weight=0, interactions=[])

                temporal_graph[source][target]['weight'] += 1
                temporal_graph[source][target]['interactions'].append(interaction)

        # Embedding do grafo social
        social_positions = nx.spring_layout(temporal_graph, dim=3)

        # Análise de comunidades
        communities = self._detect_social_communities(temporal_graph)

        # Centralidade dinâmica
        centrality = nx.betweenness_centrality(temporal_graph)

        return {
            'graph': temporal_graph,
            'embedding': social_positions,
            'communities': communities,
            'centrality': centrality,
        }

    def _calculate_manifold_curvature(self, embedding: np.ndarray):
        """Calcula curvatura Riemanniana do manifold"""
        from scipy.spatial import Delaunay

        # Triangulação do manifold
        try:
            tri = Delaunay(embedding)
        except:
            return 0.0

        curvatures = []
        for simplex in tri.simplices:
            # Pega pontos do simplex
            points = embedding[simplex]

            # Calcula curvatura local
            curvature = self._local_curvature(points)
            curvatures.append(curvature)

        return np.mean(curvatures) if curvatures else 0.0

    def _local_curvature(self, points): return 0.0

    def _cluster_behavior_states(self, states: np.ndarray):
        """Clustering de estados comportamentais"""
        try:
            from sklearn.cluster import HDBSCAN
            clusterer = HDBSCAN(
                min_cluster_size=10,
                min_samples=5,
                cluster_selection_epsilon=0.5,
            )
            labels = clusterer.fit_predict(states)
        except ImportError:
            return {}

        # Caracteriza cada cluster
        clusters = {}
        for label in np.unique(labels):
            if label == -1:
                continue

            cluster_points = states[labels == label]
            cluster_center = np.mean(cluster_points, axis=0)
            cluster_std = np.std(cluster_points, axis=0)

            clusters[f'cluster_{label}'] = {
                'center': cluster_center.tolist(),
                'std': cluster_std.tolist(),
                'size': len(cluster_points),
                'behavior_type': self._classify_behavior(cluster_center),
            }

        return clusters

    def _analyze_strategic_trajectories(self, embedding): return []
    def _detect_social_communities(self, graph): return []
    def _classify_behavior(self, center): return "unknown"
