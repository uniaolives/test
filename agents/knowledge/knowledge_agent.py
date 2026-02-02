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

class KnowledgeAgent:
    """Agente de gestão de conhecimento com RAG"""

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

    def index_document(self, doc_id: str, text: str, metadata: dict):
        """Indexa documento para recuperação"""
        if not self.encoder or not self.qdrant:
            print("⚠️ Encoder or Qdrant client not available.")
            return

        # Criar embeddings
        embedding = self.encoder.encode(text)

        # Armazenar no Qdrant
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
        """Consulta base de conhecimento"""
        if not self.encoder or not self.qdrant:
            print("⚠️ Encoder or Qdrant client not available.")
            return []

        # Embed da pergunta
        query_vector = self.encoder.encode(question)

        # Buscar similares
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
        """Gera resposta usando RAG"""
        # Recuperar contexto
        contexts = self.query(query, top_k=3)
        context_text = "\n".join([c['text'] for c in contexts])

        # Construir prompt
        prompt = f"""Contexto:
{context_text}

Pergunta: {query}

Resposta baseada no contexto:"""

        # Gerar via LLM local
        return llm_client.complete(prompt)

if __name__ == "__main__":
    agent = KnowledgeAgent()
    print("📚 Knowledge Agent initialized.")
