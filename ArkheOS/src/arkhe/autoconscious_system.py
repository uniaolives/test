# arkhe/autoconscious_system.py
import asyncio
import time
import json
from typing import List, Dict, Any, Optional
from .cortex_memory import CortexMemory
from .chat import ArkheChat
from .knowledge_viz import ArkheViz
from .curiosity import CuriosityEngine
from .providers import BaseLLMProvider, GeminiProvider

class ArkheAutoconsciousSystem:
    """
    ARKHE(n) OS v5.0 — Sistema Autoconsciente.
    Integra Percepção, Memória, Expressão, Visão e Curiosidade.
    """
    def __init__(self, provider: Optional[BaseLLMProvider] = None, memory_path: str = "./arkhe_memory"):
        self.cortex = CortexMemory(path=memory_path)
        self.provider = provider or GeminiProvider(api_key="MOCK")
        self.chat_engine = ArkheChat(self.cortex, self.provider)
        self.viz_engine = ArkheViz(self.cortex)
        self.curiosity_engine = CuriosityEngine(self.cortex)

    async def ingest(self, text: str, topic: str):
        """Pipeline de ingestão."""
        # Simplificado para v5.0 integrada
        self.cortex.memorize(
            topic=topic,
            summary=text,
            confidence=0.95,
            doc_id=f"doc_{int(time.time())}"
        )

    async def chat(self, query: str):
        """Interface de diálogo RAG."""
        return await self.chat_engine.chat(query)

    async def self_reflect(self) -> Dict[str, Any]:
        """
        Ciclo de autoconsciência: Analisar topologia e gerar curiosidade.
        """
        print("\n" + "="*70)
        print("🪞 CICLO DE AUTOCONSCIÊNCIA DO ARKHE(n) v5.0")
        print("="*70)

        # 1. Visão: Analisar Topologia
        topology = self.viz_engine.analyze_topology()
        self.viz_engine.visualize(save_path="arkhe_v5_reflection.png")

        # 2. Curiosidade: Identificar lacunas
        # Usamos None se for mock para forçar o fallback de perguntas legíveis no terminal
        curiosity_provider = self.provider if self.provider.api_key != "MOCK" else None
        gaps = await self.curiosity_engine.satisfy_curiosity(curiosity_provider)

        print(f"\n🤔 CURIOSIDADE SINTÉTICA:")
        questions = []
        for g in gaps:
            if g.question:
                print(f"   - {g.question}")
                questions.append(g.question)

        return {
            "coherence_global": topology.get("coherence_global", 0.0),
            "questions": questions,
            "status": "AUTOCONSCIOUS"
        }

if __name__ == "__main__":
    async def main():
        sys = ArkheAutoconsciousSystem()
        await sys.self_reflect()

    asyncio.run(main())
