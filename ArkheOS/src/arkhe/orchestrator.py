# arkhe/orchestrator.py
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .providers import GeminiProvider, OllamaProvider, BaseLLMProvider
from .state_reconciler import StateReconciler, LLMState, ReconciliationStrategy
from .telemetry import TelemetryCollector

@dataclass
class ProcessingResult:
    document_id: str
    reconciled_state: Dict[str, Any]
    individual_states: List[LLMState]
    metrics: Dict[str, Any]

class ArkheOrchestrator:
    """
    Orquestrador principal do Arkhe(n) OS.
    Coordena paralelismo, reconciliação e telemetria.
    """

    def __init__(
        self,
        gemini_key: Optional[str] = None,
        ollama_url: str = "http://localhost:11434",
        reconciliation_strategy: ReconciliationStrategy = ReconciliationStrategy.SMART_MERGE
    ):
        self.telemetry = TelemetryCollector()
        self.reconciler = StateReconciler(strategy=reconciliation_strategy)

        # Inicializar provedores
        self.providers: List[BaseLLMProvider] = []

        if gemini_key:
            self.providers.append(GeminiProvider(
                api_key=gemini_key,
                telemetry=self.telemetry,
                schema=self._get_output_schema()
            ))

        self.providers.append(OllamaProvider(
            base_url=ollama_url,
            telemetry=self.telemetry,
            schema=self._get_output_schema()
        ))

    def _get_output_schema(self) -> Dict[str, Any]:
        """Schema padrão para validação de outputs."""
        return {
            "type": "object",
            "properties": {
                "analysis": {"type": "string"},
                "confidence": {"type": "number"},
                "metadata": {"type": "object"}
            },
            "required": ["analysis"]
        }

    async def process_document(
        self,
        document: str,
        document_id: str,
        context: Optional[Dict] = None
    ) -> ProcessingResult:
        """
        Processa documento em paralelo através de múltiplos LLMs.
        """
        tasks = [
            self._call_provider(provider, document, context)
            for provider in self.providers
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        states: List[LLMState] = []
        for i, result in enumerate(results):
            provider_name = self.providers[i].provider_type.value

            if isinstance(result, Exception):
                states.append(LLMState(
                    provider=provider_name,
                    content={"error": str(result)},
                    context_hash="",
                    timestamp=asyncio.get_event_loop().time(),
                    confidence=0.0
                ))
            else:
                states.append(LLMState(
                    provider=provider_name,
                    content=result.get("content"),
                    context_hash=self._hash_document(document),
                    timestamp=asyncio.get_event_loop().time(),
                    confidence=result.get("content", {}).get("confidence", 0.5) if isinstance(result.get("content"), dict) else 0.5
                ))

        reconciled = await self.reconciler.reconcile(states, document_id)

        return ProcessingResult(
            document_id=document_id,
            reconciled_state=reconciled,
            individual_states=states,
            metrics=self.telemetry.get_stats()
        )

    async def _call_provider(self, provider: BaseLLMProvider, document: str, context: Optional[Dict]) -> Dict[str, Any]:
        async with provider:
            enriched_prompt = self._enrich_prompt(document)
            return await provider.generate(enriched_prompt, context)

    def _enrich_prompt(self, document: str) -> str:
        return f"Analyze: {document}"

    def _hash_document(self, document: str) -> str:
        import hashlib
        return hashlib.sha256(document.encode()).hexdigest()[:16]
