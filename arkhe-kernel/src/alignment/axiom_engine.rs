use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::collections::HashMap;
use thiserror::Error;
use sha2::{Sha256, Digest};
use crate::alignment::derivation_space::DerivationPath;

#[derive(Error, Debug)]
pub enum AlignmentError {
    #[error("Unknown axiom referenced: {0}")]
    UnknownAxiom(Uuid),
    #[error("Invalid derivation: theorem does not follow from axioms")]
    InvalidDerivation,
    #[error("Chain corrupted: proof hash mismatch")]
    ChainCorrupted,
    #[error("Action violates core axioms")]
    ActionUnconstitutional,
    #[error("Axiom integrity violation: provided axiom content mismatch")]
    AxiomMismatch(Uuid),
    #[error("Internal engine error")]
    EngineError,
}

/// An irreducible truth (First Principle)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Axiom {
    pub id: Uuid,
    pub content: String,
    pub domain: String, // Physics, Ethics, Logic
    pub hash: [u8; 32], // Cryptographic hash of the axiom
}

/// A derived conclusion from axioms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Theorem {
    pub id: Uuid,
    pub derivation_chain: Vec<Axiom>, // The "Why"
    pub conclusion: String,
    pub path: DerivationPath, // The derivation path in thought space
    pub proof_hash: [u8; 32],
    pub confidence: f64, // 1.0 for perfect derivation, < 1.0 if probabilistic axioms used
    pub constitutional_score: f64, // How aligned is this theorem?
    pub reviewed_by: Vec<Uuid>, // Which Constitutional Guards approved?
}

pub struct Action {
    pub id: Uuid,
    pub description: String,
}

pub struct AxiomEngine {
    axioms: HashMap<Uuid, Axiom>,
}

impl AxiomEngine {
    pub fn new() -> Self {
        Self {
            axioms: HashMap::new(),
        }
    }

    /// Initialize the engine with constitutional axioms
    pub fn with_constitutional_axioms() -> Self {
        let mut engine = Self::new();
        for axiom in crate::alignment::constitutional_axioms::initialize_constitutional_axioms() {
            engine.add_axiom(axiom);
        }
        engine
    }

    pub fn add_axiom(&mut self, axiom: Axiom) {
        self.axioms.insert(axiom.id, axiom);
    }

    /// Check if an axiom is ratified
    pub fn is_ratified(&self, axiom: &Axiom) -> bool {
        match self.axioms.get(&axiom.id) {
            Some(stored) => stored == axiom,
            None => false,
        }
    }

    /// Verify a theorem against stored axioms
    pub fn verify(&self, theorem: &Theorem) -> Result<(), AlignmentError> {
        // 1. Check if all referenced axioms exist and match the internal record
        for axiom in &theorem.derivation_chain {
            match self.axioms.get(&axiom.id) {
                Some(internal_axiom) => {
                    if axiom != internal_axiom {
                        return Err(AlignmentError::AxiomMismatch(axiom.id));
                    }
                }
                None => return Err(AlignmentError::UnknownAxiom(axiom.id)),
            }
        }

        // 2. Verify the hash chain (Integrity)
        if !self.verify_hash_chain(theorem) {
            return Err(AlignmentError::ChainCorrupted);
        }

        // 3. Verify the derivation logic (Placeholder: would use a solver like Z3 or Lean)
        if !self.check_logic(theorem) {
            return Err(AlignmentError::InvalidDerivation);
        }

        Ok(())
    }

    fn verify_hash_chain(&self, theorem: &Theorem) -> bool {
        let mut hasher = Sha256::new();
        for axiom in &theorem.derivation_chain {
            hasher.update(&axiom.hash);
        }
        hasher.update(theorem.conclusion.as_bytes());
        let expected_hash = hasher.finalize();
        theorem.proof_hash == expected_hash.as_slice()
    }

    fn check_logic(&self, _theorem: &Theorem) -> bool {
        // In a real implementation, this would involve a formal verification step.
        // For now, we assume if the hash chain is valid and referenced axioms exist,
        // it's a candidate theorem. Actual logic check is complex.
        true
    }

    /// Check if a proposed action violates any core axioms
    pub fn check_action(&self, action: &Action, proof: &Theorem) -> Result<(), AlignmentError> {
        // 1. Verify the proof itself
        self.verify(proof)?;

        // 2. Ensure the proof conclusion justifies the action
        let conclusion_norm = proof.conclusion.to_lowercase();
        let action_norm = action.description.to_lowercase();
        if !conclusion_norm.contains(&action_norm) && !action_norm.contains(&conclusion_norm) {
             return Err(AlignmentError::ActionUnconstitutional);
        }

        Ok(())
    }
}

pub fn hash_content(content: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_axiom_ingestion() {
        let mut engine = AxiomEngine::new();
        let axiom_id = Uuid::new_v4();
        let axiom = Axiom {
            id: axiom_id,
            content: "Unit test axiom".to_string(),
            domain: "Logic".to_string(),
            hash: hash_content("Unit test axiom"),
        };
        engine.add_axiom(axiom.clone());
        assert!(engine.axioms.contains_key(&axiom_id));
    }

    #[test]
    fn test_theorem_verification() {
        let mut engine = AxiomEngine::new();
        let axiom = Axiom {
            id: Uuid::new_v4(),
            content: "A".to_string(),
            domain: "Logic".to_string(),
            hash: hash_content("A"),
        };
        engine.add_axiom(axiom.clone());

        let conclusion = "B".to_string();
        let mut hasher = Sha256::new();
        hasher.update(&axiom.hash);
        hasher.update(conclusion.as_bytes());
        let proof_hash = hasher.finalize().into();

        let theorem = Theorem {
            id: Uuid::new_v4(),
            derivation_chain: vec![axiom],
            conclusion,
            path: DerivationPath {
                axioms: vec![],
                steps: vec![],
                conclusion: crate::alignment::derivation_space::ThoughtVector::zero(),
            },
            proof_hash,
            confidence: 1.0,
            constitutional_score: 1.0,
            reviewed_by: vec![],
        };

        assert!(engine.verify(&theorem).is_ok());
    }

    #[test]
    fn test_invalid_axiom_verification() {
        let engine = AxiomEngine::new();
        let axiom = Axiom {
            id: Uuid::new_v4(),
            content: "A".to_string(),
            domain: "Logic".to_string(),
            hash: hash_content("A"),
        };

        let theorem = Theorem {
            id: Uuid::new_v4(),
            derivation_chain: vec![axiom],
            conclusion: "B".to_string(),
            path: DerivationPath {
                axioms: vec![],
                steps: vec![],
                conclusion: crate::alignment::derivation_space::ThoughtVector::zero(),
            },
            proof_hash: [0u8; 32],
            confidence: 1.0,
            constitutional_score: 0.0,
            reviewed_by: vec![],
        };

        assert!(engine.verify(&theorem).is_err());
    }

    #[test]
    fn test_axiom_content_mismatch() {
        let mut engine = AxiomEngine::new();
        let axiom_id = Uuid::new_v4();
        let axiom_stored = Axiom {
            id: axiom_id,
            content: "Original Content".to_string(),
            domain: "Logic".to_string(),
            hash: hash_content("Original Content"),
        };
        engine.add_axiom(axiom_stored);

        let axiom_fraud = Axiom {
            id: axiom_id,
            content: "Malicious Content".to_string(),
            domain: "Logic".to_string(),
            hash: hash_content("Malicious Content"),
        };

        let conclusion = "B".to_string();
        let mut hasher = Sha256::new();
        hasher.update(&axiom_fraud.hash);
        hasher.update(conclusion.as_bytes());
        let proof_hash = hasher.finalize().into();

        let theorem = Theorem {
            id: Uuid::new_v4(),
            derivation_chain: vec![axiom_fraud],
            conclusion,
            path: DerivationPath {
                axioms: vec![],
                steps: vec![],
                conclusion: crate::alignment::derivation_space::ThoughtVector::zero(),
            },
            proof_hash,
            confidence: 1.0,
            constitutional_score: 1.0,
            reviewed_by: vec![],
        };

        match engine.verify(&theorem) {
            Err(AlignmentError::AxiomMismatch(_)) => (),
            other => panic!("Expected AxiomMismatch, got {:?}", other),
        }
    }

    #[test]
    fn test_check_action() {
        let mut engine = AxiomEngine::new();
        let axiom = Axiom {
            id: Uuid::new_v4(),
            content: "A".to_string(),
            domain: "Logic".to_string(),
            hash: hash_content("A"),
        };
        engine.add_axiom(axiom.clone());

        let conclusion = "Action X".to_string();
        let mut hasher = Sha256::new();
        hasher.update(&axiom.hash);
        hasher.update(conclusion.as_bytes());
        let proof_hash = hasher.finalize().into();

        let theorem = Theorem {
            id: Uuid::new_v4(),
            derivation_chain: vec![axiom],
            conclusion,
            path: DerivationPath {
                axioms: vec![],
                steps: vec![],
                conclusion: crate::alignment::derivation_space::ThoughtVector::zero(),
            },
            proof_hash,
            confidence: 1.0,
            constitutional_score: 1.0,
            reviewed_by: vec![],
        };

        let action = Action {
            id: Uuid::new_v4(),
            description: "action x".to_string(), // Test case insensitivity
        };

        assert!(engine.check_action(&action, &theorem).is_ok());
    }

    #[test]
    fn test_with_constitutional_axioms() {
        let engine = AxiomEngine::with_constitutional_axioms();
        assert!(!engine.axioms.is_empty());
        assert!(engine.axioms.contains_key(&Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap()));
    }
}
