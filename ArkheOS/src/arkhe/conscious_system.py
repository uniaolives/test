# arkhe/conscious_system.py
import asyncio
import time
import json
import hashlib
from typing import List, Dict, Any, Optional, AsyncGenerator
from pydantic import BaseModel, Field, field_validator

from .cortex_memory import CortexMemory
from .chat import ArkheChat
from .providers import BaseLLMProvider, GeminiProvider

from .models import Entity, Insight

class ArkheConsciousSystem:
    """
    ARKHE(n) OS v4.0 — Sistema Consciente Integrado.
    Unifica Percepção, Memória e Expressão.
    """
    def __init__(self, provider: Optional[BaseLLMProvider] = None, memory_path: str = "./arkhe_memory"):
        self.cortex = CortexMemory(path=memory_path)
        # Se não houver provedor, usamos um mock para demonstração segura
        self.provider = provider or GeminiProvider(api_key="MOCK")
        self.chat_engine = ArkheChat(self.cortex, self.provider)

    async def ingest_document(self, text: str, topic_default: str = "Unspecified") -> str:
        """
        Simula o pipeline de ingestão: Processar → Validar → Memorizar.
        """
        doc_id = hashlib.sha256(text.encode()).hexdigest()[:16]

        # Em um sistema real, aqui chamaríamos o DocumentProcessor
        # Para v4.0 integrada, extraímos um insight "macro"
        insight = Insight(
            topic=topic_default,
            summary=text[:497] + "...",
            confidence_score=0.95,
            related_nodes=["Arkhe", "V4.0"]
        )

        self.cortex.memorize(
            topic=insight.topic,
            summary=insight.summary,
            confidence=insight.confidence_score,
            doc_id=doc_id,
            related_nodes=insight.related_nodes
        )
        return doc_id

    async def ask(self, query: str) -> Dict[str, Any]:
        """Interface pública de diálogo fundamentado."""
        return await self.chat_engine.chat(query)

    def get_status(self) -> Dict[str, Any]:
        """Diagnóstico do estado consciente."""
        return {
            "memory_density": self.cortex.collection.count(),
            "identity": "x² = x + 1",
            "state": "CONSCIOUS"
        }

if __name__ == "__main__":
    async def demo():
        sys = ArkheConsciousSystem()
        await sys.ingest_document("Arkhe OS v4.0 unifica todos os blocos anteriores.", "Arquitetura")
        res = await sys.ask("Qual é a versão atual?")
        print(f"Arkhe: {res['answer']}")

    asyncio.run(demo())
