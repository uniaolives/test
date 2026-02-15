# ArkheOS Extraction Module (Π_0)
# Refactored for robust extraction, parallel LLM calls, and state reconciliation.

import asyncio
import time
import logging
import random
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Any, Dict, Type
from enum import Enum
import hashlib
from datetime import datetime, timezone

# Configure logging for telemetry
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ArkheExtraction")

class Currency(str, Enum):
    USD = "USD"
    EUR = "EUR"
    BRL = "BRL"
    GBP = "GBP"

class Provenance(BaseModel):
    """The physical anchor of an extracted fact."""
    doc_hash: str = Field(..., description="SHA256 of the source document")
    page: int = Field(..., description="Page number (1-based)")
    bbox: List[float] = Field(..., description="[x0, y0, x1, y1] coordinates")
    context_snippet: str = Field(..., description="Text snippet surrounding the fact")
    element_id: Optional[str] = Field(None, description="Link to the structural LayoutElement")
    structural_context: Optional[str] = Field(None, description="e.g. 'Table 1, Row 5, Col 3'")
    confidence: float = 0.95

class FinancialFact(BaseModel):
    """Structured financial data with geometric provenance."""
    value: float = Field(..., description="Numerical value")
    unit: Currency = Field(..., description="Currency unit")
    description: str = Field(..., description="Short description of the fact")
    confidence: float = Field(0.95, ge=0.0, le=1.0)
    provenance: Optional[Provenance] = None

class ExtractionReport(BaseModel):
    """A complete block of extracted information."""
    facts: List[FinancialFact]
    document_name: str
    extraction_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model_used: str

class BaseExtractor:
    """Base class for all extraction engines with retry and telemetry."""
    def __init__(self, model_name: str, max_retries: int = 3):
        self.model_name = model_name
        self.max_retries = max_retries

    async def _call_llm_internal(self, prompt: str) -> str:
        """To be implemented by specific LLM providers."""
        raise NotImplementedError

    async def extract(self, text: str, schema: Type[BaseModel]) -> Optional[BaseModel]:
        """Performs extraction with exponential backoff and JSON validation."""
        retries = 0
        while retries <= self.max_retries:
            start_time = time.time()
            try:
                raw_response = await self._call_llm_internal(text)
                # JSON Schema Validation via Pydantic v2
                parsed_data = schema.model_validate_json(raw_response)

                latency = time.time() - start_time
                logger.info(f"Telemetry: model={self.model_name} status=success latency={latency:.4f}s")
                return parsed_data
            except (ValidationError, Exception) as e:
                latency = time.time() - start_time
                logger.error(f"Telemetry: model={self.model_name} status=failure latency={latency:.4f}s error={type(e).__name__}")

                retries += 1
                if retries <= self.max_retries:
                    wait_time = (2 ** retries) + random.uniform(0, 1)
                    logger.warning(f"Retrying {self.model_name} in {wait_time:.2f}s... ({retries}/{self.max_retries})")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Max retries reached for model {self.model_name}")
        return None

class GeminiExtractor(BaseExtractor):
    def __init__(self, api_key: str):
        super().__init__(model_name="gemini-2.0-flash")
        self.api_key = api_key

    async def _call_llm_internal(self, prompt: str) -> str:
        # Simulated API call for architectural validation
        await asyncio.sleep(random.uniform(0.5, 1.5))
        # Returning valid JSON for simulation
        return '{"facts": [{"value": 1200000.0, "unit": "USD", "description": "simulated net profit"}], "document_name": "sim_doc", "model_used": "gemini-2.0-flash"}'

class OllamaExtractor(BaseExtractor):
    def __init__(self, base_url: str = "http://localhost:11434"):
        super().__init__(model_name="llama3")
        self.base_url = base_url

    async def _call_llm_internal(self, prompt: str) -> str:
        # Simulated Local LLM call
        await asyncio.sleep(random.uniform(1.0, 3.0))
        return '{"facts": [{"value": 50000.0, "unit": "BRL", "description": "simulated local expense"}], "document_name": "sim_doc", "model_used": "llama3"}'

class LongDocumentProcessor:
    """Handles splitting long documents and reconciling state between parallel calls."""
    def __init__(self, extractor: BaseExtractor, chunk_size: int = 2000, overlap: int = 200):
        self.extractor = extractor
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _chunk_text(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunks.append(text[i:i + self.chunk_size])
        return chunks

    async def process_document(self, text: str, doc_name: str) -> ExtractionReport:
        chunks = self._chunk_text(text)
        logger.info(f"Processing document '{doc_name}' split into {len(chunks)} chunks.")

        tasks = []
        for i, chunk in enumerate(chunks):
            # Context enrichment: adding document metadata to each chunk call
            enriched_prompt = f"Document: {doc_name}\nChunk: {i+1}/{len(chunks)}\nContent: {chunk}"
            tasks.append(self.extractor.extract(enriched_prompt, ExtractionReport))

        # Parallel LLM calls with asynchronous error handling
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.critical(f"Pipeline failure: {str(e)}")
            raise

        all_facts = []
        for i, res in enumerate(results):
            if isinstance(res, ExtractionReport):
                all_facts.extend(res.facts)
            elif isinstance(res, Exception):
                logger.error(f"Chunk {i} failed with error: {str(res)}")
            elif res is None:
                logger.error(f"Chunk {i} returned no result after retries.")

        # State Reconciliation: Deduplicate or merge overlapping facts
        # Simple reconciliation by keeping all unique facts
        reconciled_facts = self._reconcile_facts(all_facts)
        logger.info(f"Reconciliation complete for '{doc_name}': {len(all_facts)} raw facts -> {len(reconciled_facts)} unique facts.")

        return ExtractionReport(
            facts=reconciled_facts,
            document_name=doc_name,
            model_used=self.extractor.model_name
        )

    def _reconcile_facts(self, facts: List[FinancialFact]) -> List[FinancialFact]:
        """De-duplicates facts based on description and value within a window."""
        seen = set()
        unique_facts = []
        for fact in facts:
            # Create a semi-unique identifier for the fact
            fact_id = f"{fact.description}_{fact.value}_{fact.unit}"
            if fact_id not in seen:
                seen.add(fact_id)
                unique_facts.append(fact)
        return unique_facts
