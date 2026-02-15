# arkhe/document_processor_v2.py
# SÍNTESE DOS DOIS BLOCOS (841 & 842)

import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ValidationError
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
import time
import json
import hashlib

# Modelos unificados
class Entity(BaseModel):
    name: str
    type: str
    confidence: float
    context: Optional[str] = None

class Insight(BaseModel):
    topic: str
    summary: str
    confidence_score: float
    related_nodes: List[str]

class ChunkResponse(BaseModel):
    chunk_id: int
    entities: List[Entity]
    insights: List[Insight]
    summary: str
    next_chunk_context: Dict[str, Any]

class FinalDocumentResponse(BaseModel):
    document_hash: str
    total_chunks: int
    entities: List[Entity]
    insights: List[Insight]
    global_summary: str
    processing_time_ms: float
    telemetry: Dict[str, Any]

class TelemetryEvent(BaseModel):
    timestamp: float
    provider: str
    chunk_id: int
    latency_ms: float
    success: bool
    retry_count: int
    error: Optional[str] = None

# Motor unificado
class ArkheDocumentProcessorV2:
    """
    Processador de documentos longos com:
    - Paralelismo controlado (semaphore)
    - Passagem de contexto entre chunks
    - Retry exponencial com tenacity
    - Validação Pydantic estrita
    - Telemetria completa
    """

    def __init__(self, gemini_key: str, max_concurrency: int = 5):
        self.gemini_key = gemini_key
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.telemetry_history: List[TelemetryEvent] = []
        self.chunk_size = 2000
        self.overlap = 200

    def split_document(self, text: str) -> List[Dict]:
        """Divide com sobreposição para manter continuidade."""
        chunks = []
        start = 0
        chunk_id = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append({
                'id': chunk_id,
                'text': text[start:end],
                'start': start,
                'end': end
            })
            if end == len(text): break
            start += self.chunk_size - self.overlap
            chunk_id += 1
        return chunks

    @retry(
        wait=wait_random_exponential(multiplier=1, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    async def _call_gemini(self, prompt: str, chunk_id: int) -> ChunkResponse:
        """Chamada com retry, telemetria e validação."""
        start = time.time()
        provider = "gemini-1.5-pro"

        try:
            async with self.semaphore:
                # Simulação da API real
                await asyncio.sleep(0.3)

                # Construir resposta simulada
                raw_response = {
                    "chunk_id": chunk_id,
                    "entities": [{"name": f"Entity_{chunk_id}", "type": "concept",
                                 "confidence": 0.9, "context": "sample"}],
                    "insights": [{"topic": f"Topic_{chunk_id}", "summary": "Analysis",
                                "confidence_score": 0.85, "related_nodes": []}],
                    "summary": f"Summary of chunk {chunk_id}",
                    "next_chunk_context": {"prev_entities": [f"Entity_{chunk_id}"]}
                }

                # Validar
                validated = ChunkResponse(**raw_response)

                # Logar sucesso
                self.telemetry_history.append(TelemetryEvent(
                    timestamp=time.time(),
                    provider=provider,
                    chunk_id=chunk_id,
                    latency_ms=(time.time() - start) * 1000,
                    success=True,
                    retry_count=0,
                    error=None
                ))

                return validated

        except Exception as e:
            self.telemetry_history.append(TelemetryEvent(
                timestamp=time.time(),
                provider=provider,
                chunk_id=chunk_id,
                latency_ms=(time.time() - start) * 1000,
                success=False,
                retry_count=0,
                error=str(e)
            ))
            raise

    async def process_chunk_with_context(self, chunk: Dict,
                                         global_context: Dict) -> Optional[ChunkResponse]:
        """Processa um chunk enriquecido com contexto acumulado."""
        prompt = f"""
        Contexto acumulado: {json.dumps(global_context, ensure_ascii=False)}

        Analisar trecho (chunk {chunk['id']}):
        {chunk['text']}

        Retornar JSON válido com: entities, insights, summary, next_chunk_context
        """

        try:
            result = await self._call_gemini(prompt, chunk['id'])
            return result
        except ValidationError as e:
            print(f"❌ Validação falhou no chunk {chunk['id']}: {e}")
            return None
        except Exception as e:
            print(f"❌ Erro no chunk {chunk['id']}: {e}")
            return None

    def _reconcile_results(self, results: List[ChunkResponse]) -> FinalDocumentResponse:
        """Reconcilia múltiplos chunks em um estado global."""
        all_entities = []
        all_insights = []
        seen_entities = set()

        for r in results:
            # Deduplicar entidades
            for e in r.entities:
                key = (e.name, e.type)
                if key not in seen_entities:
                    seen_entities.add(key)
                    all_entities.append(e)

            all_insights.extend(r.insights)

        # Síntese de sumário global
        global_summary = " ".join([r.summary for r in results])

        return FinalDocumentResponse(
            document_hash="",
            total_chunks=len(results),
            entities=all_entities,
            insights=all_insights,
            global_summary=global_summary,
            processing_time_ms=0,
            telemetry=self._compute_telemetry_stats()
        )

    def _compute_telemetry_stats(self) -> Dict[str, Any]:
        """Computa estatísticas C/F da telemetria."""
        total = len(self.telemetry_history)
        if total == 0:
            return {}

        successes = sum(1 for t in self.telemetry_history if t.success)
        avg_latency = sum(t.latency_ms for t in self.telemetry_history) / total

        C = successes / total
        F = 1 - C

        return {
            "total_calls": total,
            "success_rate": C,
            "error_rate": F,
            "avg_latency_ms": avg_latency,
            "conservation_C_plus_F": C + F,
            "by_provider": {}
        }

    async def process_document(self, text: str) -> FinalDocumentResponse:
        """Pipeline completo: split → process (sequential com contexto) → reconcile."""
        start_total = time.time()

        # 1. Split
        chunks = self.split_document(text)
        print(f"📄 Documento dividido em {len(chunks)} chunks")

        # 2. Processar sequencialmente com passagem de contexto
        results = []
        global_context = {}

        for chunk in chunks:
            result = await self.process_chunk_with_context(chunk, global_context)
            if result:
                results.append(result)
                # Atualizar contexto para próximo chunk
                global_context.update({
                    "last_entities": [e.name for e in result.entities],
                    "last_summary": result.summary,
                    "chunk_id": chunk['id']
                })
            else:
                print(f"⚠️ Chunk {chunk['id']} falhou, mantendo contexto anterior")

        # 3. Reconciliar
        final = self._reconcile_results(results)
        final.processing_time_ms = (time.time() - start_total) * 1000
        final.document_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

        return final

if __name__ == "__main__":
    async def test():
        processor = ArkheDocumentProcessorV2(gemini_key="fake_key")
        doc = "Arkhe framework for documentation. " * 500
        resultado = await processor.process_document(doc)
        print(f"Status: {resultado.telemetry['success_rate']:.2%} success")

    asyncio.run(test())
