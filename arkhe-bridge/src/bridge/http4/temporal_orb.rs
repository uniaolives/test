use super::eigenstate::TemporalEigenstate;
use super::confinement::QuantumWell;
use super::methods::ConfinementMode;
use super::temporal_schrodinger::TemporalSchrodinger;
use anyhow::Result;

pub struct TemporalOrb {
    pub data: Vec<u8>,
    pub lambda_2: f64,
    pub eigenstates: Vec<TemporalEigenstate>,
}

impl TemporalOrb {
    pub fn new(data: Vec<u8>, lambda_2: f64) -> Self {
        Self {
            data,
            lambda_2,
            eigenstates: Vec::new(),
        }
    }

    /// Applies "quantum confinement" logic to determine eigenstates
    pub fn confine(&mut self, well: &QuantumWell) -> Result<()> {
        let mode = ConfinementMode::from_lambda2(self.lambda_2);

        self.eigenstates = match mode {
            ConfinementMode::InfiniteWell => vec![
                TemporalEigenstate::new(1, "GROUND", 1.0, "GROUND_ANCHORED")
            ],
            ConfinementMode::FiniteWell | ConfinementMode::Barrier => {
                TemporalSchrodinger::solve(well, 3)?
            },
            ConfinementMode::Free => vec![
                TemporalEigenstate::new(1, "DECOHERENT", 0.1, "FREE")
            ],
        };

        Ok(())
    }

    /// Calculate retrocausal tunneling probability
    pub fn retrocausal_tunneling_probability(&self, delta_t_secs: f64) -> f64 {
        let barrier_width = delta_t_secs.abs();

        // Effective barrier is modulated by coherence (lambda-2)
        // High coherence reduces the effective width of the temporal barrier
        let effective_barrier = (1.0 - self.lambda_2) * barrier_width;

        // T ~ exp(-2 * effective_barrier)
        (-2.0 * effective_barrier).exp()
    }
}
