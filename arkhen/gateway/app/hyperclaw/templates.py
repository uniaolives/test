# arkhen/gateway/app/hyperclaw/templates.py
from typing import Dict, Any, List
from pydantic import BaseModel

class HyperClawTemplate(BaseModel):
    id: str
    name: str
    description: str
    goals: Dict[str, float]
    module_configs: List[Dict[str, Any]]

BIOTECH_TEMPLATES = [
    HyperClawTemplate(
        id="gene-therapy-auto",
        name="Gene Therapy Manufacturing Automation",
        description="Fast-loop orchestration for automated protein layer recording and viral vector optimization.",
        goals={"stability": 0.9, "yield": 0.8, "purity": 0.95},
        module_configs=[
            {"name": "bio_sensor_hub", "capability": "sensor_integration"},
            {"name": "optimization_llm", "capability": "parameter_refinement"}
        ]
    ),
    HyperClawTemplate(
        id="clinical-trial-consortium",
        name="Clinical Trial Data Consortiums",
        description="Slow-loop strategic coordination for cross-institutional data validation and privacy-preserving synthesis.",
        goals={"privacy": 1.0, "interoperability": 0.85, "speed": 0.7},
        module_configs=[
            {"name": "zk_proof_executor", "capability": "privacy_validation"},
            {"name": "federated_learner", "capability": "model_aggregation"}
        ]
    ),
    HyperClawTemplate(
        id="longevity-biomarkers",
        name="Longevity Biomarker Standardization",
        description="DMR-driven monitoring of t_KR across heterogeneous longitudinal datasets to identify aging bifurcations.",
        goals={"accuracy": 0.95, "consistency": 0.9},
        module_configs=[
            {"name": "dmr_analyzer", "capability": "history_reconstruction"},
            {"name": "biomarker_llm", "capability": "trait_correlation"}
        ]
    ),
    HyperClawTemplate(
        id="bci-realtime",
        name="BCI Real-time Entrainment",
        description="High-frequency coupling between neural telemetry and synthetic oscillators using Kuramoto-based synchrony.",
        goals={"latency": 0.99, "coherence": 0.9},
        module_configs=[
            {"name": "neural_interface", "capability": "telemetry_stream"},
            {"name": "kuramoto_node", "capability": "oscillator_sync"}
        ]
    )
]
