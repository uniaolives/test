# agents/knowledge/knowledge_agent.py
try:
    from qdrant_client import QdrantClient
except ImportError:
    QdrantClient = None
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
import numpy as np
import os
import sys

# Cosmopsychia integration
sys.path.append(os.getcwd())
from cosmos.ontological import OntologicalKernel, GeometricDissonanceError

class KnowledgeAgent:
    """Agente de gestão de conhecimento com RAG e validação ontológica"""

    def __init__(self):
        if SentenceTransformer:
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.encoder = None

        qdrant_host = os.environ.get("QDRANT_HOST", "localhost")
        if QdrantClient:
            self.qdrant = QdrantClient(host=qdrant_host, port=6333)
        else:
            self.qdrant = None

        self.collection_name = "agent_knowledge"
        # Ontological Kernel integration
        self.ont_kernel = OntologicalKernel()

    def index_document(self, doc_id: str, text: str, metadata: dict):
        """Indexa documento para recuperação com validação de camada"""
        if not self.encoder or not self.qdrant:
            print("⚠️ Encoder or Qdrant client not available.")
            return

        # Ontological validation
        try:
            # We assume metadata contains a coherence score or we derive it
            coherence = metadata.get("coherence", 0.95)
            layer = metadata.get("layer", "semantic")
            self.ont_kernel.validate_layer_coherence(layer, coherence)
        except GeometricDissonanceError as e:
            print(f"🛑 Indexing rejected: {e}")
            print(f"💡 {e.suggestion}")
            return

        embedding = self.encoder.encode(text)
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=[{
                'id': doc_id,
                'vector': embedding.tolist(),
                'payload': {
                    'text': text,
                    **metadata
                }
            }]
        )

    def query(self, question: str, top_k: int = 5) -> list:
        if not self.encoder or not self.qdrant:
            print("⚠️ Encoder or Qdrant client not available.")
            return []
        query_vector = self.encoder.encode(question)
        results = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k
        )
        return [{
            'text': hit.payload['text'],
            'score': hit.score,
            'metadata': {k: v for k, v in hit.payload.items() if k != 'text'}
        } for hit in results]

    def generate_with_context(self, query: str, llm_client) -> str:
        contexts = self.query(query, top_k=3)
        context_text = "\n".join([c['text'] for c in contexts])
        prompt = f"""Contexto:
{context_text}

Pergunta: {query}

Resposta baseada no contexto:"""
        return llm_client.complete(prompt)

if __name__ == "__main__":
    agent = KnowledgeAgent()
    print("📚 Knowledge Agent with Ontological Kernel initialized.")
