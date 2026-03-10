use super::eigenstate::TemporalEigenstate;
use super::confinement::QuantumWell;
use super::methods::ConfinementMode;
use super::temporal_schrodinger::TemporalSchrodinger;
use anyhow::Result;
use super::methods::ConfinementMode;

pub struct TemporalOrb {
    pub data: Vec<u8>,
    pub lambda_2: f64,
    pub eigenstates: Vec<TemporalEigenstate>,
}

#[derive(Debug, Clone)]
pub struct TemporalEigenstate {
    pub n: u32,
    pub probability: f64,
    pub mode: String,
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
    pub fn confine(&mut self) {
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
                TemporalEigenstate { n: 1, probability: 1.0, mode: "GROUND_ANCHORED".to_string() }
            ],
            ConfinementMode::FiniteWell => vec![
                TemporalEigenstate { n: 1, probability: 0.9, mode: "GROUND_EXCITED".to_string() },
                TemporalEigenstate { n: 2, probability: 0.1, mode: "FIRST_EXCITED".to_string() }
            ],
            ConfinementMode::Barrier => vec![
                TemporalEigenstate { n: 1, probability: 0.7, mode: "TUNNELING".to_string() },
                TemporalEigenstate { n: 2, probability: 0.3, mode: "VIRTUAL".to_string() }
            ],
            ConfinementMode::Free => vec![
                TemporalEigenstate { n: 1, probability: 0.1, mode: "DECOHERING".to_string() }
            ],
        };
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
