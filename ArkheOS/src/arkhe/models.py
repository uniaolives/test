# arkhe/models.py
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime, timezone

class Entity(BaseModel):
    name: str
    type: str
    confidence: float = Field(ge=0.0, le=1.0)
    context: Optional[str] = None

class Insight(BaseModel):
    topic: str
    summary: str = Field(max_length=1000)
    confidence_score: float = Field(ge=0.0, le=1.0)
    related_nodes: List[str] = Field(default_factory=list)
    source_chunk: Optional[int] = None

class Currency(str, Enum):
    USD = "USD"
    EUR = "EUR"
    BRL = "BRL"
    GBP = "GBP"

class FinancialFact(BaseModel):
    value: float
    unit: Currency
    description: str
    confidence: float = 0.95

class ExtractionReport(BaseModel):
    facts: List[FinancialFact] = Field(default_factory=list)
    entities: List[Entity] = Field(default_factory=list)
    insights: List[Insight] = Field(default_factory=list)
    summary: str = ""
    document_name: str # Mandatory for tests
    extraction_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model_used: str # Mandatory for tests
    next_chunk_context: Dict[str, Any] = Field(default_factory=dict)

class FinalDocumentResponse(BaseModel):
    document_hash: str
    total_chunks: int
    entities: List[Entity]
    insights: List[Insight]
    global_summary: str
    processing_time_ms: float
    telemetry: Dict[str, Any]
