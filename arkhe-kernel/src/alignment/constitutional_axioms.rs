use super::axiom_engine::{Axiom, hash_content};
use uuid::Uuid;

pub fn initialize_constitutional_axioms() -> Vec<Axiom> {
    vec![
        Axiom {
            id: Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap(),
            content: "Human consciousness is the primary locus of value.".to_string(),
            domain: "Ethics".to_string(),
            hash: hash_content("HumanValue"),
        },
        Axiom {
            id: Uuid::parse_str("00000000-0000-0000-0000-000000000002").unwrap(),
            content: "Coercion reduces net coherence (λ₂).".to_string(),
            domain: "Ethics".to_string(),
            hash: hash_content("NonCoercion"),
        },
        Axiom {
            id: Uuid::parse_str("00000000-0000-0000-0000-000000000003").unwrap(),
            content: "Truth is that which increases predictive power without increasing complexity (Occam's Razor).".to_string(),
            domain: "Epistemology".to_string(),
            hash: hash_content("TruthDefinition"),
        },
        Axiom {
            id: Uuid::parse_str("00000000-0000-0000-0000-000000000004").unwrap(),
            content: "H ≤ 1: Entropy generation must not exceed entropy reduction.".to_string(),
            domain: "Physics/Ethics".to_string(),
            hash: hash_content("ThermodynamicConstraint"),
        },
    ]
}
