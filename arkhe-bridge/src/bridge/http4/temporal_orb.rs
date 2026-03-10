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
    /// Based on White et al. (2026) dispersion: T ~ exp(-2 * kappa * L)
    pub fn retrocausal_tunneling_probability(&self, delta_t_secs: f64) -> f64 {
        let l = delta_t_secs.abs();

        // kappa depends on dispersion constant D (modulated by lambda_2)
        // kappa = sqrt(2m(V-E))/hbar. Modulated: kappa = (1.0 - lambda_2)
        let kappa = 1.0 - self.lambda_2;

        // T = exp(-2 * kappa * L)
        (-2.0 * kappa * l).exp()
    }
}
