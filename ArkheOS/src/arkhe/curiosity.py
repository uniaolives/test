# arkhe/curiosity.py
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .cortex_memory import CortexMemory

@dataclass
class KnowledgeGap:
    """Representa uma lacuna de conhecimento detectada."""
    centroid: np.ndarray  # Ponto no espaço de embeddings
    radius: float          # Raio da região esparsa
    density: float         # Densidade local (menor = mais lacuna)
    description: str = ""  # Descrição da lacuna
    question: str = ""     # Pergunta gerada para preencher a lacuna

class CuriosityEngine:
    """
    Motor de curiosidade sintética.
    Identifica lacunas no espaço semântico e gera perguntas.
    """
    def __init__(self, cortex: CortexMemory, min_density_threshold: float = 0.3):
        self.cortex = cortex
        self.min_density_threshold = min_density_threshold
        self.gaps: List[KnowledgeGap] = []
        self.curiosity_level = 0.5  # F controlado

    def detect_gaps(self) -> List[KnowledgeGap]:
        """
        Analisa o espaço de embeddings e identifica regiões de baixa densidade.
        """
        # 1. Obter todos os embeddings do córtex
        data = self.cortex.collection.get(include=['embeddings', 'metadatas'])
        all_embeddings = np.array(data['embeddings'])
        metadatas = data['metadatas']

        if len(all_embeddings) < 5:
            return []

        # 2. Calcular densidade local com KNN
        k = min(10, len(all_embeddings) - 1)
        nbrs = NearestNeighbors(n_neighbors=k)
        nbrs.fit(all_embeddings)
        distances, indices = nbrs.kneighbors(all_embeddings)

        # Densidade ~ inverso da distância média aos vizinhos
        mean_dists = np.mean(distances, axis=1)
        densities = 1.0 / (mean_dists + 1e-6)

        # Normalizar densidades [0, 1]
        if densities.max() > densities.min():
            densities = (densities - densities.min()) / (densities.max() - densities.min())

        # 3. Identificar pontos de baixa densidade
        low_density_mask = densities < self.min_density_threshold
        low_density_points = all_embeddings[low_density_mask]

        if len(low_density_points) == 0:
            return []

        # 4. Clusterizar pontos de baixa densidade para identificar lacunas
        # Usamos um eps razoável para o espaço de embeddings (cosseno/L2)
        clustering = DBSCAN(eps=0.5, min_samples=2).fit(low_density_points)
        labels = clustering.labels_

        gaps = []
        for label in set(labels):
            if label == -1: continue # Ruído

            cluster_points = low_density_points[labels == label]
            centroid = np.mean(cluster_points, axis=0)
            radius = float(np.max(np.linalg.norm(cluster_points - centroid, axis=1)))
            avg_density = float(np.mean(densities[low_density_mask][labels == label]))

            # Identificar tópico próximo para descrição
            dist_to_centroid = np.linalg.norm(all_embeddings - centroid, axis=1)
            closest_idx = np.argmin(dist_to_centroid)
            closest_topic = metadatas[closest_idx].get('topic', 'Unknown')

            gap = KnowledgeGap(
                centroid=centroid,
                radius=radius,
                density=avg_density,
                description=f"Região próxima a: {closest_topic}"
            )
            gaps.append(gap)

        self.gaps = gaps
        return gaps

    async def generate_questions(self, gap: KnowledgeGap, provider=None) -> List[str]:
        """
        Gera perguntas para explorar uma lacuna.
        """
        prompt = (
            f"Como um sistema Arkhe(n) OS, identifiquei uma lacuna de conhecimento na seguinte área:\n"
            f"{gap.description}\n\n"
            f"Gere 2 perguntas investigativas que ajudem a preencher este vazio semântico, "
            f"conectando-o à identidade x² = x + 1."
        )

        if provider:
            response = await provider.generate(prompt, validate_output=False)
            questions = response.get('content', '').split('\n')
            return [q.strip() for q in questions if q.strip() and '?' in q][:2]

        # Fallback
        return [f"Quais são os limites da relação entre {gap.description} e a totalidade?"]

    async def satisfy_curiosity(self, provider=None):
        """Ciclo completo de curiosidade."""
        gaps = self.detect_gaps()
        for gap in gaps:
            questions = await self.generate_questions(gap, provider)
            if questions:
                gap.question = questions[0]

        # Aumentar nível de curiosidade proporcional aos vazios
        if gaps:
            self.curiosity_level = min(1.0, self.curiosity_level + 0.1 * len(gaps))

        return gaps

if __name__ == "__main__":
    from .cortex_memory import CortexMemory
    import asyncio

    async def test():
        cortex = CortexMemory()
        engine = CuriosityEngine(cortex)
        gaps = engine.detect_gaps()
        print(f"Detectadas {len(gaps)} lacunas.")
        for g in gaps:
            print(f"- {g.description} (Densidade: {g.density:.2f})")

    # asyncio.run(test())
