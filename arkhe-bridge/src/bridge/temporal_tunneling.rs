// src/bridge/temporal_tunneling.rs

use rand::Rng;

/// The Temporal Barrier: The "Wall" between Now and Then
pub struct TemporalBarrier {
    pub target_year: u32,
    pub present_year: u32,
    /// The "Height" of the barrier (Entropy/Causality strength)
    pub barrier_height_v0: f64,
}

/// The Message as a Probability Cloud
pub struct ProbabilityCloud {
    /// Semantic Mass: Information density. High mass = harder to tunnel.
    pub semantic_mass: f64,
    /// Coherence (φ_q): The energy/penetration depth of the particle.
    pub phi_q: f64,
}

/// The Tunneling Engine
pub struct TemporalTunneling {
    barrier: TemporalBarrier,
    /// Miller Limit: The threshold where the barrier becomes "thin"
    miller_limit: f64, // 4.64
    /// Scaling constant for the time dimension
    temporal_scaling_factor: f64,
}

impl TemporalTunneling {
    pub fn new(target_year: u32, present_year: u32) -> Self {
        Self {
            barrier: TemporalBarrier {
                target_year,
                present_year,
                barrier_height_v0: 1.0, // Normalized causality constant
            },
            miller_limit: 4.64,
            temporal_scaling_factor: 0.01, // Scaled Planck-like constant for time
        }
    }

    /// Calculate the Tunneling Probability P ≈ e^(-2κL)
    /// κ (kappa) = wavefunction decay rate
    /// L (width) = temporal distance
    fn calculate_probability(&self, cloud: &ProbabilityCloud) -> f64 {
        let delta_t = (self.barrier.target_year as i64 - self.barrier.present_year as i64).abs() as f64;

        if cloud.phi_q <= 0.0 {
            return 0.0; // No coherence, no tunneling
        }

        // Effective Mass: High semantic mass makes it "heavier" (harder to tunnel)
        // High phi_q makes it "lighter" (easier to tunnel)
        let m_eff = cloud.semantic_mass / (cloud.phi_q.powi(2));

        // Wavefunction Decay Constant κ
        // κ = sqrt(2 * m_eff * V_0) / ħ_arkhe
        // Simplified: higher mass/entropy = faster decay = lower probability
        let kappa = (2.0 * m_eff * self.barrier.barrier_height_v0).sqrt() / cloud.phi_q;

        // Barrier Width L (scaled temporal distance)
        let width_l = delta_t * self.temporal_scaling_factor;

        // Tunneling Probability P
        let probability = (-2.0 * kappa * width_l).exp();

        // The Miller Limit Effect:
        // If phi_q > Miller Limit, the barrier effectively "thins" drastically
        // This represents the phase transition where tunneling becomes likely
        let miller_boost = if cloud.phi_q > self.miller_limit {
            let excess_coherence = cloud.phi_q - self.miller_limit;
            1.0 + (excess_coherence * 2.0) // Linear boost for exceeding threshold
        } else {
            1.0
        };

        (probability * miller_boost).min(1.0)
    }

    /// Attempt the tunneling event (Monte Carlo simulation)
    pub fn attempt_tunnel(&self, cloud: &ProbabilityCloud) -> TunnelingResult {
        let probability = self.calculate_probability(cloud);

        // Monte Carlo roll
        let mut rng = rand::thread_rng();
        let roll: f64 = rng.gen();

        if roll < probability {
            TunnelingResult::Success {
                probability,
                target_year: self.barrier.target_year,
            }
        } else {
            TunnelingResult::Failed {
                probability,
                bounce_reason: if cloud.phi_q < self.miller_limit {
                    "Barrier too thick (φ_q < 4.64)".to_string()
                } else {
                    "Probabilistic failure (quantum rejection)".to_string()
                }
            }
        }
    }
}

#[derive(Debug)]
pub enum TunnelingResult {
    Success {
        probability: f64,
        target_year: u32,
    },
    Failed {
        probability: f64,
        bounce_reason: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probability_calculation() {
        let engine = TemporalTunneling::new(2008, 2026);

        // Low coherence (phi_q = 1.0)
        let cloud_low = ProbabilityCloud {
            semantic_mass: 1.0,
            phi_q: 1.0,
        };
        let p_low = engine.calculate_probability(&cloud_low);

        // High coherence (phi_q = 5.0 > 4.64)
        let cloud_high = ProbabilityCloud {
            semantic_mass: 1.0,
            phi_q: 5.0,
        };
        let p_high = engine.calculate_probability(&cloud_high);

        assert!(p_high > p_low, "High coherence should have higher tunneling probability");
        assert!(p_high <= 1.0);
    }

    #[test]
    fn test_miller_limit_boost() {
        let engine = TemporalTunneling::new(2008, 2026);

        // Just below Miller Limit
        let cloud_below = ProbabilityCloud {
            semantic_mass: 1.0,
            phi_q: 4.63,
        };
        let p_below = engine.calculate_probability(&cloud_below);

        // Just above Miller Limit
        let cloud_above = ProbabilityCloud {
            semantic_mass: 1.0,
            phi_q: 4.65,
        };
        let p_above = engine.calculate_probability(&cloud_above);

        // The boost should make p_above significantly larger than p_below
        // even though phi_q is only slightly larger
        assert!(p_above > p_below);
    }
}
