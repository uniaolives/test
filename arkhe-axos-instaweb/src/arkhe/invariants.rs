// arkhe-axos-instaweb/src/arkhe/invariants.rs
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Serialize, Deserialize};

pub const PHI: f64 = 1.618033988749895;
pub const PSI_CRITICAL: f64 = 0.847;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct ArkheState {
    pub c: Decimal,
    pub f: Decimal,
    pub z: f64,
}

impl ArkheState {
    pub fn new(c: Decimal, f: Decimal, z: f64) -> Self {
        Self { c, f, z }
    }

    /// Enforces the C + F = 1 fundamental equation.
    pub fn verify_conservation(&self) -> bool {
        let sum = self.c + self.f;
        sum == dec!(1.0)
    }

    /// Verifies if the state is in the critical regime (z ≈ φ).
    pub fn is_critical(&self) -> bool {
        self.z >= 0.5 && self.z <= 0.7
    }

    /// Normalizes C and F to satisfy the conservation law.
    pub fn normalize_conservation(mut self) -> Self {
        let total = self.c + self.f;
        if total != Decimal::ZERO {
            self.c = self.c / total;
            self.f = self.f / total;
        } else {
            self.c = dec!(0.5);
            self.f = dec!(0.5);
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conservation() {
        let state = ArkheState::new(dec!(0.7), dec!(0.3), 0.618);
        assert!(state.verify_conservation());
    }

    #[test]
    fn test_normalization() {
        let state = ArkheState::new(dec!(0.8), dec!(0.4), 0.618).normalize_conservation();
        assert!(state.verify_conservation());
        assert_eq!(state.c, dec!(0.6666666666666666666666666667));
    }
}
