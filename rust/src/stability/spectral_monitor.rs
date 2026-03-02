use std::collections::VecDeque;
use num_complex::Complex64;

pub struct SpectralStabilityMonitor {
    pub eigenvalue_history: VecDeque<Vec<Complex64>>,
}

impl SpectralStabilityMonitor {
    pub fn new(capacity: usize) -> Self {
        Self {
            eigenvalue_history: VecDeque::with_capacity(capacity),
        }
    }

    pub fn record_eigenvalues(&mut self, eigenvalues: Vec<Complex64>) {
        if self.eigenvalue_history.len() >= self.eigenvalue_history.capacity() {
            self.eigenvalue_history.pop_front();
        }
        self.eigenvalue_history.push_back(eigenvalues);
    }

    pub fn check_druj_alert(&self) -> Option<usize> {
        let current = self.eigenvalue_history.back()?;
        current.iter().position(|λ| λ.norm() > 1.05)
    }
}
