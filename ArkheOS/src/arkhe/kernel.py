# ArkheOS Structural Kernel (Π_3)
# The Physics of the Document

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import hashlib

@dataclass
class LayoutElement:
    """A fundamental stone in the document's physics."""
    id: str
    type: str  # "paragraph", "table", "cell", "header", "word"
    text: str
    bbox: List[float]  # [x0, y0, x1, y1] normalized or absolute
    page: int
    metadata: Dict[str, Any] = field(default_factory=dict)

class DocumentIngestor:
    """
    The Kernel that processes the 'physics' of the file.
    Supports Local parsing and prepared for Cloud OCR (Azure/AWS).
    """
    def __init__(self, provider="local"):
        self.provider = provider

    def process(self, file_path: str) -> List[LayoutElement]:
        """Converts a document into a list of structured layout elements."""
        elements = []
        if self.provider == "local":
            # Simulation of local structural parsing using pdfplumber logic
            # In a real scenario, this would iterate through words and tables
            elements.append(LayoutElement(
                id="p1_w1",
                type="word",
                text="Profit",
                bbox=[100, 400, 150, 420],
                page=1
            ))
            elements.append(LayoutElement(
                id="p1_t1",
                type="table",
                text="Income Statement Table",
                bbox=[50, 300, 500, 600],
                page=1,
                metadata={"rows": 10, "cols": 5}
            ))
        return elements

class AnchorResolver:
    """
    Maps an extracted fact back to the physical OCR element.
    Ensures the "Geodesic" anchor is load-bearing.
    """
    @staticmethod
    def find_best_anchor(target_text: str, elements: List[LayoutElement]) -> Optional[LayoutElement]:
        """Finds the most likely physical anchor for a given text using Jaccard similarity."""
        best_score = 0.0
        best_el = None

        target_tokens = set(target_text.lower().split())
        for el in elements:
            el_tokens = set(el.text.lower().split())
            if not el_tokens:
                continue

            # Jaccard Similarity
            score = len(target_tokens & el_tokens) / len(target_tokens | el_tokens)
            if score > best_score:
                best_score = score
                best_el = el

        # Return best match if above a confidence threshold
        return best_el if best_score > 0.5 else None
