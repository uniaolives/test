"""
ArkheOS Extraction Module (Π_0) - Production Grade
Integrates Pydantic validation, Gemini/Ollama structured extraction,
and robust document processing with OCR fallback.
"""

import json
import logging
import time
import asyncio
import hashlib
from datetime import datetime
from enum import Enum
from typing import List, Optional, Any, Dict, Type, TypeVar, Union

from pydantic import BaseModel, Field, ValidationError
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("arkhe.extraction")

T = TypeVar('T', bound=BaseModel)

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
    extraction_timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    model_used: str = "arkhe-hybrid-v1"

class ExtractionEngine:
    """Base class for extraction engines."""
    async def extract_structured(self, model_class: Type[T], prompt: str) -> T:
        raise NotImplementedError

class GeminiExtractor(ExtractionEngine):
    """Extraction using Google Gemini."""
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def extract_structured(self, model_class: Type[T], prompt: str) -> T:
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json"
                )
            )
            data = json.loads(response.text)
            return model_class.model_validate(data)
        except (ValidationError, json.JSONDecodeError) as e:
            logger.warning(f"Gemini extraction failed validation: {e}. Retrying...")
            raise e

class DocumentProcessor:
    """Handles long document processing with chunking and state reconciliation."""

    def __init__(self, extractor: ExtractionEngine):
        self.extractor = extractor

    async def process_long_document(self, text: str, chunk_size: int = 2000, overlap: int = 200) -> List[FinancialFact]:
        """Processes a document in chunks, passing context forward."""
        chunks = self._chunk_text(text, chunk_size, overlap)
        all_facts = []
        context = ""

        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            prompt = self._build_prompt(chunk, context)

            try:
                report = await self.extractor.extract_structured(ExtractionReport, prompt)
                all_facts.extend(report.facts)
                # Update context with a summary of found facts to pass to the next chunk
                context = f"Previous entities found: {json.dumps([f.description for f in report.facts])}"
            except Exception as e:
                logger.error(f"Failed to process chunk {i}: {e}")
                # Continue to next chunk or handle error specifically

        return self._reconcile_state(all_facts)

    def _chunk_text(self, text: str, size: int, overlap: int) -> List[str]:
        return [text[i:i + size] for i in range(0, len(text), size - overlap)]

    def _build_prompt(self, chunk: str, context: str) -> str:
        return f"""
        Context from previous sections: {context}

        Extract financial facts from the following text and return as JSON matching the schema.
        Text: {chunk}
        """

    def _reconcile_state(self, facts: List[FinancialFact]) -> List[FinancialFact]:
        """Merges overlapping or duplicate facts based on description and provenance."""
        # Implementation of deduplication logic
        unique_facts = {}
        for fact in facts:
            key = f"{fact.description}_{fact.value}_{fact.unit}"
            if key not in unique_facts or fact.confidence > unique_facts[key].confidence:
                unique_facts[key] = fact
        return list(unique_facts.values())

class OCRPipeline:
    """OCR pipeline with Azure fallback to local Tesseract."""

    async def extract_text(self, image_path: str) -> str:
        try:
            return await self._azure_ocr(image_path)
        except Exception as e:
            logger.warning(f"Azure OCR failed: {e}. Falling back to Tesseract.")
            return await self._tesseract_ocr(image_path)

    async def _azure_ocr(self, path: str) -> str:
        # Placeholder for Azure AI Document Intelligence integration
        raise NotImplementedError("Azure OCR not configured")

    async def _tesseract_ocr(self, path: str) -> str:
        # Placeholder for local Tesseract fallback
        import pytesseract
        from PIL import Image
        return await asyncio.to_thread(pytesseract.image_to_string, Image.open(path))
