// src/gateway/tiger_payment_header.rs

use zeroize::{Zeroize, ZeroizeOnDrop};
use crate::payments::gkp_vault::GKPBalance;

#[derive(Debug, Clone, Zeroize, ZeroizeOnDrop)]
pub struct PaymentHeaderP0 {
    pub estimated_cost: u128,
    pub payment_nonce: [u8; 32],
    pub pre_balance: GKPBalance,
}

impl PaymentHeaderP0 {
    pub fn validate_quantum_proof(&self) -> Result<(), String> {
        // Mock verification
        Ok(())
    }
}

// Implement Zeroize manually for GKPBalance if not already derived
impl Zeroize for GKPBalance {
    fn zeroize(&mut self) {
        self.quantum_state.zeroize();
        self.amplitude = num_complex::Complex64::new(0.0, 0.0);
        self.uncertainty = 0.0;
    }
}
