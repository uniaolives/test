# explainability_bridge.py
# Memory ID 41-A: Manifold-to-Legal Translation Layer

import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger("ExplainabilityBridge")

class ExplainabilityBridge:
    """
    Translates HDC gradients and 659D manifold geometry
    into human-readable SASC Attestations and legal justifications.
    """

    def __init__(self, hdc_engine):
        self.hdc_engine = hdc_engine
        # The 'Rosetta' mapping: subspace indices to CF/88 and UDHR articles
        self.legal_rosetta = self._load_legal_rosetta()
        self.attestation_history = []

    async def generate_justification(self, state_vector: torch.Tensor, hdc_score: float, current_phi: float = 0.60) -> Dict[str, Any]:
        """
        Main entry point: generates a human-readable justification for an HDC state.
        """
        # 1. Perform Attribution Analysis (Sensitivity Analysis)
        attributions = self._compute_feature_attributions(state_vector)

        # 2. Map Attributions to Legal Articles
        primary_drivers = self._map_to_legal_basis(attributions)

        # 3. Generate Natural Language Explanation (Simplified)
        explanation = self._synthesize_explanation(hdc_score, primary_drivers)

        # 4. Create Attestation
        attestation = {
            'attestation_id': f"CRUX-86-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'phi_at_event': current_phi,
            'hdc_score': hdc_score,
            'legal_basis': primary_drivers,
            'explanation': explanation,
            'causal_path_hash': self._hash_causal_path(state_vector),
            'liability_allocation': self._compute_liability_weights(hdc_score)
        }

        self.attestation_history.append(attestation)
        return attestation

    def _compute_feature_attributions(self, state_vector: torch.Tensor) -> torch.Tensor:
        """
        Simplified attribution: compute contribution of each subspace to deviation from 1.0.
        In production, this would use Integrated Gradients.
        """
        # For simulation, we look at how far each element is from 'ideal' (assumed 1.0 for sigmoid)
        # We focus on the mean of each subspace
        attributions = torch.zeros_like(state_vector)
        for name, mask in self.hdc_engine.subspace_masks.items():
            subspace_vals = torch.sigmoid(state_vector[mask])
            # If value is low, it contributes more to the 'drop'
            contribution = 1.0 - subspace_vals.mean().item()
            attributions[mask] = contribution / (mask.stop - mask.start)
        return attributions

    def _map_to_legal_basis(self, attributions: torch.Tensor) -> List[Dict[str, Any]]:
        relevant_articles = []
        for subspace, mask in self.hdc_engine.subspace_masks.items():
            # Calculate the contribution of this subspace
            contribution = torch.sum(attributions[mask]).item()
            if contribution > 0.10:  # Significance threshold
                article = self.legal_rosetta.get(subspace)
                relevant_articles.append({
                    'subspace': subspace,
                    'article': article['ref'],
                    'description': article['desc'],
                    'severity': float(contribution)
                })
        return sorted(relevant_articles, key=lambda x: x['severity'], reverse=True)

    def _synthesize_explanation(self, hdc_score: float, drivers: List[Dict]) -> str:
        if hdc_score >= 0.95:
            return "Ação em plena conformidade com os princípios da dignidade humana."

        if not drivers:
            return "Queda marginal no HDC sem violação sistêmica detectada."

        primary = drivers[0]
        return f"Veto preventivo acionado por risco em {primary['subspace']}. Base legal: {primary['article']} ({primary['description']}). Severidade detectada: {primary['severity']:.2f}."

    def _compute_liability_weights(self, hdc_score: float) -> Dict[str, float]:
        # Default distribution from MID-41
        base = {'creator': 0.45, 'sasc': 0.30, 'agent': 0.25}
        if hdc_score < 0.90:
            base['agent'] += 0.10
            base['creator'] -= 0.10
        return base

    def _hash_causal_path(self, state_vector: torch.Tensor) -> str:
        import hashlib
        return hashlib.sha256(state_vector.detach().numpy().tobytes()).hexdigest()[:16]

    def _load_legal_rosetta(self) -> Dict[str, Dict[str, str]]:
        return {
            'autonomy': {'ref': 'CF/88 Art. 1º, III', 'desc': 'Dignidade e Autonomia Individual'},
            'integrity': {'ref': 'CF/88 Art. 5º, caput', 'desc': 'Inviolabilidade do Direito à Vida e Segurança'},
            'privacy': {'ref': 'LGPD / CF/88 Art. 5º, X', 'desc': 'Privacidade e Proteção de Dados'},
            'equity': {'ref': 'CF/88 Art. 3º, IV', 'desc': 'Promoção do Bem de Todos sem Preconceitos'}
        }
