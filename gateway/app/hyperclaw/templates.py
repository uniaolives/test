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
    },
    "at_home_diagnostics": {
        "goals": {"false_positive_rate": 0.01, "user_compliance": 0.9, "data_encryption_level": 1.0},
        "mode": Mode.VALIDATE,
        "budget": {"compute": 1000, "time": 3600}
    },
    "offshore_reg_infra": {
        "goals": {"jurisdictional_compliance": 1.0, "audit_transparency": 0.95, "latency_to_mainland": 50.0},
        "mode": Mode.WRAP_UP,
        "budget": {"compute": 5000, "time": 86400}
    },
    "in_vivo_gene_editing": {
        "goals": {"off_target_effect": 0.001, "transduction_efficiency": 0.8, "immune_response_index": 0.1},
        "mode": Mode.REFINE,
        "budget": {"compute": 8000, "time": 1209600}
    },
    "cellular_reprogramming": {
        "goals": {"pluripotency_markers": 0.98, "differentiation_rate": 0.9, "genomic_stability": 1.0},
        "mode": Mode.EXPLORE,
        "budget": {"compute": 6000, "time": 2419200}
    },
    "continuous_metabolic": {
        "goals": {"glucose_variance": 0.05, "insulin_sensitivity_index": 0.8, "prediction_horizon_min": 30.0},
        "mode": Mode.REFINE,
        "budget": {"compute": 2000, "time": 3600}
    },
    "regenerative_med": {
        "goals": {"tissue_regrowth_mm": 5.0, "vascularization_depth": 0.7, "stem_cell_viability": 0.95},
        "mode": Mode.EXPLORE,
        "budget": {"compute": 7000, "time": 1814400}
    },
    "synthetic_biology": {
        "goals": {"circuit_fidelity": 0.99, "metabolic_load_balance": 0.8, "yield_per_liter": 10.0},
        "mode": Mode.EXPLORE,
        "budget": {"compute": 5000, "time": 604800}
    },
    "biomanufacturing_infra": {
        "goals": {"uptime_percentage": 0.999, "batch_cycle_time": 48.0, "waste_reduction_index": 0.4},
        "mode": Mode.WRAP_UP,
        "budget": {"compute": 10000, "time": 86400}
    },
    "personalized_medicine": {
        "goals": {"treatment_efficacy_score": 0.9, "side_effect_probability": 0.05, "genomic_fit_index": 0.95},
        "mode": Mode.REFINE,
        "budget": {"compute": 9000, "time": 172800}
    },
    "engineered_microbiome": {
        "goals": {"species_diversity_index": 0.8, "metabolite_production": 0.5, "host_symbiosis_score": 0.9},
        "mode": Mode.EXPLORE,
        "budget": {"compute": 3000, "time": 604800}
    },
    "ruminant_genetics_ovis": {
        "goals": {
            "krt_fiber_optimization": 0.9,
            "rumen_metabolic_efficiency": 0.85,
            "karyotype_stability_2n54": 1.0,
            "transgenic_expression_yield": 0.75
        },
        "mode": Mode.EXPLORE,
        "budget": {"compute": 4500, "time": 1209600}
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
