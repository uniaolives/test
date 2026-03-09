# rag/pipeline.py
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient

class ArkheRAG:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.client = QdrantClient(url="http://qdrant:6333")

    def retrieve_context(self, query: str, top_k: int = 5):
        # 1. Busca Vetorial
        results = self.client.search(
            collection_name="teknet_memory",
            query_vector=self.embeddings.embed_query(query),
            limit=top_k,
            query_filter={
                "must": [
                    {"key": "coherence", "range": {"gte": 0.5}} # Apenas coerente
                ]
            }
        )
        return results
