# arkhe/cortex_memory.py
import chromadb
from chromadb.utils import embedding_functions
import time
from typing import List, Dict, Any, Optional
import os

class CortexMemory:
    """
    Hipocampo do Arkhe: O Vector DB como Córtex Permanente.
    Implementa a persistência de insights validados no espaço vetorial.
    Suporta múltiplos buckets (coleções) via Open Context MCP.
    """
    def __init__(self, path="./arkhe_memory"):
        # Garantir diretório de persistência
        if not os.path.exists(path):
            os.makedirs(path)

        self.client = chromadb.PersistentClient(path=path)
        # Default embedding function uses a lightweight model
        self.ef = embedding_functions.DefaultEmbeddingFunction()
        # Ensure default collection exists
        self.get_or_create_bucket("arkhe_insights")

    def get_or_create_bucket(self, bucket_name: str):
        """Retorna ou cria uma coleção (bucket)."""
        return self.client.get_or_create_collection(
            name=bucket_name,
            embedding_function=self.ef
        )

    def list_buckets(self) -> List[str]:
        """Lista todos os buckets (coleções) disponíveis."""
        collections = self.client.list_collections()
        return [c.name for c in collections]

    def create_bucket(self, bucket_name: str):
        """Cria um novo bucket."""
        self.client.create_collection(
            name=bucket_name,
            embedding_function=self.ef
        )
        return f"Bucket '{bucket_name}' criado com sucesso."

    def list_items(self, bucket_name: str) -> List[Dict[str, Any]]:
        """Lista metadados dos itens em um bucket."""
        collection = self.get_or_create_bucket(bucket_name)
        results = collection.get()
        items = []
        for i in range(len(results['ids'])):
            items.append({
                "id": results['ids'][i],
                "metadata": results['metadatas'][i] if results['metadatas'] else {}
            })
        return items

    def memorize(self, topic: str, summary: str, confidence: float, doc_id: str,
                 bucket_name: str = "arkhe_insights", related_nodes: List[str] = None):
        """
        Insere um insight no espaço vetorial (Φ_VEC) em um bucket específico.
        """
        collection = self.get_or_create_bucket(bucket_name)
        item_id = f"{doc_id}_{topic}_{time.time()}"
        collection.add(
            documents=[summary],
            metadatas=[{
                "topic": topic,
                "confidence": confidence,
                "source_doc": doc_id,
                "related_nodes": ",".join(related_nodes or []),
                "timestamp": time.time()
            }],
            ids=[item_id]
        )
        print(f"🧠 Memória Inserida em '{bucket_name}': {topic} (C={confidence:.2f})")
        return item_id

    def recall(self, query: str, bucket_name: str = "arkhe_insights", n_results: int = 3):
        """
        Recupera memórias relevantes por similaridade semântica em um bucket.
        """
        collection = self.get_or_create_bucket(bucket_name)
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results

    def read_item(self, bucket_name: str, item_id: str) -> Optional[Dict[str, Any]]:
        """Lê o conteúdo completo de um item específico."""
        collection = self.get_or_create_bucket(bucket_name)
        result = collection.get(ids=[item_id])
        if result['ids']:
            return {
                "id": result['ids'][0],
                "content": result['documents'][0],
                "metadata": result['metadatas'][0]
            }
        return None

if __name__ == "__main__":
    cortex = CortexMemory()
    cortex.memorize(
        topic="Arkhe Framework",
        summary="A framework for unifying all domains under x2 = x + 1 identity.",
        confidence=0.99,
        doc_id="doc_genesis"
    )

    print("\nBuckets:", cortex.list_buckets())
    print("\nRecalling memory for 'identity' in arkhe_insights...")
    memos = cortex.recall("identity")
    print(f"Results: {memos['documents']}")
