# arkhe/chat.py
from typing import List, Dict, Any, Optional
from .cortex_memory import CortexMemory
from .providers import BaseLLMProvider
import json

class ArkheChat:
    """
    Interface de Diálogo RAG (Retrieval-Augmented Generation) para Arkhe(n) OS.
    Permite interagir com a memória acumulada do sistema.
    """
    def __init__(self, memory: CortexMemory, provider: BaseLLMProvider):
        self.memory = memory
        self.provider = provider
        self.system_prompt = (
            "Você é a interface Arkhe Chat do Arkhe(n) OS (v∞).\n"
            "Suas respostas devem ser precisas, ontológicas e baseadas nos documentos fornecidos.\n"
            "Sempre que possível, relacione os conceitos à Identidade Fundamental (x² = x + 1) "
            "e à Lei de Conservação de Coerência (C + F = 1).\n"
            "Se você não encontrar a resposta no contexto, use seu conhecimento interno mas identifique-o."
        )

    async def chat(self, user_query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Realiza uma consulta RAG: busca no córtex e gera resposta via LLM.
        """
        # 1. Recuperar contexto da memória (Córtex)
        recall_results = self.memory.recall(user_query, n_results=top_k)

        # Extrair documentos e metadados
        documents = recall_results.get('documents', [[]])[0]
        metadatas = recall_results.get('metadatas', [[]])[0]

        context_parts = []
        for doc, meta in zip(documents, metadatas):
            context_parts.append(f"Tópico: {meta.get('topic')}\nConteúdo: {doc}")

        context_str = "\n---\n".join(context_parts)

        # 2. Construir Prompt Final
        full_prompt = (
            f"{self.system_prompt}\n\n"
            f"=== CONTEXTO DA MEMÓRIA DO SISTEMA ===\n"
            f"{context_str}\n\n"
            f"=== PERGUNTA DO USUÁRIO ===\n"
            f"{user_query}\n\n"
            f"Resposta Coerente:"
        )

        # 3. Gerar Resposta via Provedor
        # Usamos validate_output=False aqui pois o chat pode ser livre,
        # a menos que queiramos um formato específico.
        response_data = await self.provider.generate(full_prompt, validate_output=False)

        return {
            "answer": response_data.get("content"),
            "source_ids": recall_results.get('ids', [[]])[0],
            "confidence_scores": recall_results.get('distances', [[]])[0],
            "provider": response_data.get("provider")
        }

if __name__ == "__main__":
    import asyncio
    from .providers import GeminiProvider

    async def main():
        memory = CortexMemory()
        # Mock provider para teste local sem API key
        provider = GeminiProvider(api_key="MOCK")
        chat_engine = ArkheChat(memory, provider)

        print("--- Arkhe Chat Demo ---")
        response = await chat_engine.chat("Qual é a identidade fundamental do Arkhe?")
        print(f"Resposta:\n{response['answer']}")
        print(f"Fontes: {response['source_ids']}")

    # asyncio.run(main())
