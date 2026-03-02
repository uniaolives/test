// interest_service.rs
// Implementation of Mainframe-equivalent arithmetic in Rust
// Part of the ASI-Ω Legacy Modernization Suite

use rust_decimal::prelude::*;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct TransactionInput {
    pub principal: Decimal, // Mapped from PIC 9(9)V99
    pub rate: Decimal,      // Mapped from PIC 9(3)V99
    pub periods: i32,       // Mapped from PIC 9(3)
}

pub struct InterestEngine;

impl InterestEngine {
    /// Calculates simple interest with the same truncation/rounding behavior as COBOL
    /// Rule: INTEREST = PRINCIPAL * (RATE / 100) * PERIODS
    pub fn calculate_fixed_interest(input: TransactionInput) -> Decimal {
        let divisor = dec!(100.0);

        // In COBOL, operation order and rounding are fixed by PICTURE clauses and ROUNDED verb.
        // We simulate 'ROUNDED' behavior or standard financial truncation.
        let rate_factor = (input.rate / divisor).round_dp(4); // Intermediate precision
        let interest = (input.principal * rate_factor * Decimal::from(input.periods))
            .round_dp(2); // Final rounding to 2 decimal places (the penny)

        interest
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_penny_equivalence() {
        let input = TransactionInput {
            principal: dec!(10000.00),
            rate: dec!(5.25),
            periods: 1,
        };

        let result = InterestEngine::calculate_fixed_interest(input);

        // Floating point would give 524.999999999 or similar
        // Decimal guarantees exactly 525.00
        assert_eq!(result, dec!(525.00));
    }
}
