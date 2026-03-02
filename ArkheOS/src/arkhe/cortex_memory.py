# arkhe/cortex_memory.py
import chromadb
from chromadb.utils import embedding_functions
import time
from typing import List, Dict, Any
import os

class CortexMemory:
    """
    Hipocampo do Arkhe: O Vector DB como Córtex Permanente.
    Implementa a persistência de insights validados no espaço vetorial.
    """
    def __init__(self, path="./arkhe_memory"):
        # Garantir diretório de persistência
        if not os.path.exists(path):
            os.makedirs(path)

        self.client = chromadb.PersistentClient(path=path)
        # Default embedding function uses a lightweight model
        self.ef = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name="arkhe_insights",
            embedding_function=self.ef
        )

    def memorize(self, topic: str, summary: str, confidence: float, doc_id: str, related_nodes: List[str] = None):
        """
        Insere um insight no espaço vetorial (Φ_VEC).
        """
        self.collection.add(
            documents=[summary],
            metadatas=[{
                "topic": topic,
                "confidence": confidence,
                "source_doc": doc_id,
                "related_nodes": ",".join(related_nodes or [])
            }],
            ids=[f"{doc_id}_{topic}_{time.time()}"]
        )
        print(f"🧠 Memória Inserida: {topic} (C={confidence:.2f})")

    def recall(self, query: str, n_results: int = 3):
        """
        Recupera memórias relevantes por similaridade semântica.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results

if __name__ == "__main__":
    cortex = CortexMemory()
    cortex.memorize(
        topic="Arkhe Framework",
        summary="A framework for unifying all domains under x2 = x + 1 identity.",
        confidence=0.99,
        doc_id="doc_genesis"
    )

    print("\nRecalling memory for 'identity'...")
    memos = cortex.recall("identity")
    print(f"Results: {memos['documents']}")
