//! bio_sequence.rs
//! Stubs for biological sequence primitives used by MetaConsciousness

use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RNASequence {
    pub bases: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecondaryStructure {
    pub dot_bracket: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethylationPattern {
    pub sites: Vec<usize>,
}
