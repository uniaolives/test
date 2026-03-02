pub use crate::sasc::*;
pub use crate::attestation::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainId {
    Mainnet,
    Testnet,
    Sovereign,
}

pub struct SASCValidator;

impl SASCValidator {
    pub fn new() -> Self {
        Self
    }
}
