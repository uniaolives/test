use std::sync::atomic::{AtomicU64, Ordering};
use crate::security::invariant_engine::GateError;

pub enum ComplexityClass {
    Low,
    Medium,
    High,
    Shor2048,
}

pub struct Gate8MultiverseRegulator {
    pub current_allocation: AtomicU64,
    pub max_allocation: u64,
}

impl Gate8MultiverseRegulator {
    pub fn new(max_alloc: u64) -> Self {
        Self {
            current_allocation: AtomicU64::new(0),
            max_allocation: max_alloc,
        }
    }

    pub fn authorize_computation(&self, complexity: ComplexityClass) -> Result<(), GateError> {
        let required_universes = match complexity {
            ComplexityClass::Low => 1_000,
            ComplexityClass::Medium => 1_000_000,
            ComplexityClass::High => 1_000_000_000,
            ComplexityClass::Shor2048 => 1_000_000_000_000_000,
        };

        let current = self.current_allocation.load(Ordering::SeqCst);
        if current + required_universes > self.max_allocation {
            return Err(GateError::Gate8Failure);
        }

        self.current_allocation.fetch_add(required_universes, Ordering::SeqCst);
        Ok(())
    }
}
