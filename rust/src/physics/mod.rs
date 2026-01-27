#[derive(Debug, Clone)]
pub struct VonNeumannEntropy(pub f64);

impl VonNeumannEntropy {
    pub fn new(v: f64) -> Self {
        Self(v)
    }
    pub fn value(&self) -> f64 {
        self.0
    }
    pub fn coherence_collapse_detected(&self, drift: f64) -> bool {
        self.0 < drift
    }
}

pub struct CoherenceMonitor;
