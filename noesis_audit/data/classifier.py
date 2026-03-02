# noesis-audit/data/classifier.py
"""
Classificação contínua de dados sensíveis para agentes NOESIS.
"""

import re
from typing import List, Dict, Tuple, Generator

class DataClassifier:
    """
    Classifica automaticamente todos os dados que entram no ecossistema.
    Integra com padrões conhecidos de PII, PHI e segredos corporativos.
    """

    SENSITIVE_PATTERNS = {
        "PII": r"\b\d{3}-\d{2}-\d{4}\b",  # SSN (exemplo)
        "PHI": r"patient_[0-9]+",         # Saúde
        "FINANCIAL": r"card_\d{16}",      # Cartão de crédito
        "PROPRIETARY": r"patent_[A-Z0-9]+", # Propriedade intelectual
        "SECRET_KEY": r"(?i)API[_-]KEY[:\s]+[A-Z0-9]{32,}" # Chaves de API
    }

    def classify_stream(self, data_stream: List[str]) -> Generator[Tuple[str, List[str]], None, None]:
        for record in data_stream:
            tags = []
            for category, pattern in self.SENSITIVE_PATTERNS.items():
                if re.search(pattern, str(record)):
                    tags.append(category)
            yield (record, tags)

    def analyze_text(self, text: str) -> List[str]:
        tags = []
        for category, pattern in self.SENSITIVE_PATTERNS.items():
            if re.search(pattern, text):
                tags.append(category)
        return tags
