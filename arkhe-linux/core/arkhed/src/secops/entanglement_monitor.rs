use arkhe_quantum::QuantumState;
use rand::Rng;

pub struct EntanglementMonitor {
    pairs: Vec<EntangledPair>,
    chsh_threshold: f64,
}

pub struct EntangledPair {
    pub node_a: [u8; 8],
    pub node_b: [u8; 8],
    pub state: QuantumState,
}

impl EntanglementMonitor {
    pub fn new() -> Self {
        Self {
            pairs: Vec::new(),
            chsh_threshold: 2.0,
        }
    }

    pub fn add_pair(&mut self, pair: EntangledPair) {
        self.pairs.push(pair);
    }

    pub fn run_test(&self) -> Vec<EntanglementAlert> {
        let mut alerts = Vec::new();
        let mut rng = rand::thread_rng();

        for pair in &self.pairs {
            // Simulated CHSH test using QuantumState entropy as a proxy for entanglement quality
            let entropy = pair.state.von_neumann_entropy();

            // In a real system, we'd do measurements in different bases.
            // Here, we simulate the S parameter (CHSH) based on entropy.
            // S_max = 2√2 ≈ 2.828. If entropy is high, S drops towards 2.0 (classical limit).
            let base_s = 2.828;
            let chsh_estimate = base_s - (entropy * 1.5) + rng.gen_range(-0.1..0.1);

            if chsh_estimate < self.chsh_threshold {
                alerts.push(EntanglementAlert {
                    node_a: pair.node_a,
                    node_b: pair.node_b,
                    reason: format!("CHSH violation: S = {:.3} (Entropy: {:.3})", chsh_estimate, entropy),
                    chsh_value: chsh_estimate,
                });
            }
        }
        alerts
    }
}

pub struct EntanglementAlert {
    pub node_a: [u8; 8],
    pub node_b: [u8; 8],
    pub reason: String,
    pub chsh_value: f64,
}
