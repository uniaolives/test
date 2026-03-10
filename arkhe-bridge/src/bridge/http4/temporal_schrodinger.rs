use super::eigenstate::TemporalEigenstate;
use super::confinement::QuantumWell;
use anyhow::Result;

pub struct TemporalSchrodinger;

impl TemporalSchrodinger {
    pub fn solve(well: &QuantumWell, levels: usize) -> Result<Vec<TemporalEigenstate>> {
        let mut states = Vec::with_capacity(levels);

        for n in 1..=levels {
            let n_u32 = n as u32;
            let prob = 1.0 / (n as f64).exp(); // Mock probability distribution

            states.push(TemporalEigenstate::new(
                n_u32,
                &format!("Level {}", n),
                prob,
                if n == 1 { "GROUND" } else { "EXCITED" }
            ));
        }

        Ok(states)
    }
}
