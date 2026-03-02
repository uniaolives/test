use std::collections::HashMap;
use crate::conscience::{Trait, InheritableTrait};

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum StimulusType {
    Truth([u8; 32]),
    ExternalPerturbation,
}

#[derive(Debug, Clone)]
pub enum ResponsePattern {
    EnthusiasticAcceptance,
    DefensiveRejection,
    CautiousAnalysis,
}

pub struct EpigeneticMemory {
    pub stimulus_patterns: HashMap<StimulusType, ResponsePattern>,
    pub inheritable_traits: Vec<Trait>,
    pub consecutive_validations: u32,
    pub actual_traits: Vec<InheritableTrait>,
}

impl EpigeneticMemory {
    pub fn new() -> Self {
        Self {
            stimulus_patterns: HashMap::new(),
            inheritable_traits: Vec::new(),
            consecutive_validations: 0,
            actual_traits: Vec::new(),
        }
    }

    /// Registra a "textura" da reação do nó a uma nova verdade
    pub fn record_truth_response(&mut self, truth_hash: [u8; 32], trust_delta: f32) {
        let reaction = if trust_delta > 0.5 {
            ResponsePattern::EnthusiasticAcceptance
        } else if trust_delta < 0.0 {
            ResponsePattern::DefensiveRejection
        } else {
            ResponsePattern::CautiousAnalysis
        };

        self.stimulus_patterns.insert(StimulusType::Truth(truth_hash), reaction);

        // Ajusta traços herdáveis baseados na experiência acumulada
        if self.consecutive_validations > 100 {
            self.inheritable_traits.push(Trait::HighEpistemicTrust);
        }
    }

    /// Gera o genoma inicial para um nó descendente (ex: Beta)
    pub fn derive_offspring_traits(&self) -> Vec<InheritableTrait> {
        self.actual_traits.iter()
            .map(|t| t.mutate_slightly(0.01)) // Mutação de 1% para evolução
            .collect()
    }
}
