from .loops import ContextFrame, Mode

BIOTECH_TEMPLATES = {
    "gene_therapy_automation": {
        "goals": {"batch_consistency": 0.95, "purity_threshold": 0.99, "yield_optimization": 0.8},
        "mode": Mode.REFINE,
        "budget": {"compute": 5000, "time": 86400}
    },
    "clinical_trial_consortium": {
        "goals": {"data_privacy": 1.0, "cross_site_correlation": 0.85, "patient_safety_compliance": 1.0},
        "mode": Mode.VALIDATE,
        "budget": {"compute": 10000, "time": 604800}
    },
    "longevity_standardization": {
        "goals": {"biomarker_reliability": 0.9, "inter_lab_calibration": 0.8, "biological_age_variance": 0.1},
        "mode": Mode.EXPLORE,
        "budget": {"compute": 3000, "time": 2592000}
    },
    "bci_realtime_entrainment": {
        "goals": {"latency_ms": 10.0, "classification_accuracy": 0.95, "neuroplasticity_index": 0.7},
        "mode": Mode.EXPLORE,
        "budget": {"compute": 2000, "time": 3600}
    },
    "mitochondrial_enhancement": {
        "goals": {"atp_production_gain": 0.2, "mt_dna_integrity": 0.99, "oxidative_stress_reduction": 0.85},
        "mode": Mode.REFINE,
        "budget": {"compute": 4000, "time": 432000}
    }
}

def spawn_frame_from_template(template_id: str, dmr_id: str = "default") -> ContextFrame:
    if template_id not in BIOTECH_TEMPLATES:
        return ContextFrame(dmr_id=dmr_id)

    t = BIOTECH_TEMPLATES[template_id]
    return ContextFrame(
        goals=t["goals"],
        mode=t["mode"],
        budget=t["budget"],
        dmr_id=dmr_id
    )
