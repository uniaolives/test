# arkhe/document_processor_v2.py
# ARKHE(n) v5.0 — Parallel Document Processor

import asyncio
import time
import json
import hashlib
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ValidationError

from .models import Entity, Insight, ExtractionReport, FinalDocumentResponse
from .providers import GeminiProvider
from .telemetry import Provider, TelemetryCollector

class ArkheDocumentProcessorV2:
    """
    Processador de documentos longos com:
    - Paralelismo controlado (semaphore)
    - Reconciliação de estado global
    - Telemetria integrada via TelemetryCollector
    - Validação Pydantic
    """

    def __init__(self, gemini_key: str, max_concurrency: int = 5, telemetry: Optional[TelemetryCollector] = None):
        self.telemetry = telemetry or TelemetryCollector()
        self.provider = GeminiProvider(api_key=gemini_key, telemetry=self.telemetry)
        self.semaphore = asyncio.Semaphore(max_concurrency)
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

    async def process_chunk(self, chunk: Dict, doc_hash: str) -> Optional[ExtractionReport]:
        """Processa um chunk individualmente."""
        prompt = f"""
        Analisar trecho do documento (Hash: {doc_hash}, Chunk: {chunk['id']}):
        {chunk['text']}

        Retornar JSON válido satisfazendo o schema ExtractionReport.
        """

        try:
            async with self.semaphore:
                # O provedor já lida com retries e telemetria internamente se configurado
                response_data = await self.provider.generate(prompt, context={"model_used": "gemini-1.5-pro"}, validate_output=False)
                content = response_data.get("content")

                # Parsing e Validação
                raw_json = json.loads(content)
                # Garantir campos obrigatórios para o schema
                raw_json.setdefault("document_name", doc_hash)
                raw_json.setdefault("model_used", "gemini-1.5-pro")

                return ExtractionReport(**raw_json)
        except Exception as e:
            print(f"❌ Falha no chunk {chunk['id']}: {e}")
            return None

    def _reconcile_results(self, results: List[ExtractionReport], doc_hash: str) -> FinalDocumentResponse:
        """Reconcilia múltiplos chunks em um estado global."""
        all_entities = []
        all_insights = []
        seen_entities = set()
        summaries = []

        for r in results:
            # Deduplicar entidades
            for e in r.entities:
                key = (e.name.lower(), e.type)
                if key not in seen_entities:
                    seen_entities.add(key)
                    all_entities.append(e)

            all_insights.extend(r.insights)
            if r.summary:
                summaries.append(r.summary)

        # Síntese de sumário global (simples concatenação ou lógica mais complexa)
        global_summary = " ".join(summaries)

        return FinalDocumentResponse(
            document_hash=doc_hash,
            total_chunks=len(results),
            entities=all_entities,
            insights=all_insights,
            global_summary=global_summary,
            processing_time_ms=0, # Set by caller
            telemetry=self.telemetry.get_stats(Provider.GEMINI)
        )

    async def process_document(self, text: str) -> FinalDocumentResponse:
        """Pipeline completo PARALELO."""
        start_total = time.time()

        doc_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        chunks = self.split_document(text)
        print(f"📄 Processando {doc_hash} ({len(chunks)} chunks) em paralelo...")

        # Criar tarefas para processamento paralelo
        tasks = [self.process_chunk(c, doc_hash) for c in chunks]

        # Executar em paralelo
        raw_results = await asyncio.gather(*tasks)

        # Filtrar sucessos
        valid_results = [r for r in raw_results if r is not None]

        # Reconciliar
        final = self._reconcile_results(valid_results, doc_hash)
        final.processing_time_ms = (time.time() - start_total) * 1000

        return final

if __name__ == "__main__":
    async def test():
        # Usar MOCK key para simulação
        processor = ArkheDocumentProcessorV2(gemini_key="MOCK")
        doc = "Arkhe framework content. " * 100
        resultado = await processor.process_document(doc)
        print(f"✅ Processado em {resultado.processing_time_ms:.0f}ms")
        print(f"   Entidades: {len(resultado.entities)}")
        print(f"   Insights: {len(resultado.insights)}")

    asyncio.run(test())
