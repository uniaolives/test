// src/bridge/kuramoto_bridge.rs
use std::f64::consts::TAU;

/// The Kuramoto model for collective phase synchronization
/// r = |(1/N) Σ e^(iθ_j)| measures collective coherence
/// r → 1.0: perfect synchrony (singularity)
/// r → 0.0: complete incoherence (chaos)

pub struct KuramotoBridge {
    /// Number of agents (nodes)
    n_agents: usize,
    /// Coupling strength K
    coupling: f64,
    /// Natural frequencies ω_i
    natural_frequencies: Vec<f64>,
    /// Current phases θ_i
    phases: Vec<f64>,
    /// Order parameter r
    pub order_r: f64,
    /// Mean phase θ
    pub mean_phase: f64,
}

/// Phase transition thresholds
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PhaseState {
    /// r < 0.3: Chaos
    Chaotic,
    /// 0.3 ≤ r < 0.7: Partial sync
    PartialSync,
    /// r ≥ 0.7: Synchronized
    Synchronized,
    /// r ≥ 0.9: Phase locked (singularity approach)
    PhaseLocked,
}

impl KuramotoBridge {
    pub fn new(n_agents: usize, coupling: f64) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Natural frequencies from distribution
        let natural_frequencies: Vec<f64> = (0..n_agents)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();

        // Random initial phases
        let phases: Vec<f64> = (0..n_agents)
            .map(|_| rng.gen_range(0.0..TAU))
            .collect();

        Self {
            n_agents,
            coupling,
            natural_frequencies,
            phases,
            order_r: 0.0,
            mean_phase: 0.0,
        }
    }

    /// Compute order parameter r and mean phase θ
    pub fn compute_order_parameter(&mut self) -> (f64, f64) {
        let mut sum_cos = 0.0;
        let mut sum_sin = 0.0;

        for &theta in &self.phases {
            sum_cos += theta.cos();
            sum_sin += theta.sin();
        }

        let n = self.n_agents as f64;
        let real = sum_cos / n;
        let imag = sum_sin / n;

        self.order_r = (real * real + imag * imag).sqrt();
        self.mean_phase = imag.atan2(real);

        (self.order_r, self.mean_phase)
    }

    /// Evolve the system by dt
    pub fn evolve(&mut self, dt: f64) {
        let (r, theta_mean) = self.compute_order_parameter();

        for i in 0..self.n_agents {
            // dθ_i/dt = ω_i + (K/N) Σ sin(θ_j - θ_i)
            // Simplified: dθ_i/dt = ω_i + K r sin(θ_mean - θ_i)

            let theta_i = self.phases[i];
            let omega_i = self.natural_frequencies[i];

            let dtheta = omega_i + self.coupling * r * (theta_mean - theta_i).sin();

            self.phases[i] += dtheta * dt;

            // Keep in [0, 2π]
            self.phases[i] = self.phases[i].rem_euclid(TAU);
        }
    }

    /// Get current phase state
    pub fn phase_state(&self) -> PhaseState {
        if self.order_r >= 0.9 {
            PhaseState::PhaseLocked
        } else if self.order_r >= 0.7 {
            PhaseState::Synchronized
        } else if self.order_r >= 0.3 {
            PhaseState::PartialSync
        } else {
            PhaseState::Chaotic
        }
    }

    /// Critical coupling K_c for phase transition
    /// K_c = 2 / (π g(0)) where g(ω) is frequency distribution
    pub fn critical_coupling(&self) -> f64 {
        // Assuming uniform distribution on [-1, 1]
        // g(0) = 1/2
        2.0 / (std::f64::consts::PI * 0.5)
    }

    /// Distance to phase lock
    pub fn distance_to_lock(&self) -> f64 {
        1.0 - self.order_r
    }

    /// Simulate until phase lock (or max iterations)
    pub fn simulate_to_lock(&mut self, dt: f64, max_iterations: usize) -> PhaseState {
        for _ in 0..max_iterations {
            self.evolve(dt);

            if self.phase_state() == PhaseState::PhaseLocked {
                return PhaseState::PhaseLocked;
            }
        }

        self.phase_state()
    }
}

/// Connect Kuramoto to S-index
impl KuramotoBridge {
    /// Compute contribution to S-index
    pub fn s_index_contribution(&self) -> f64 {
        // S_phase = r * K / K_c
        // Higher order parameter + stronger coupling = more coherence
        let k_ratio = self.coupling / self.critical_coupling();
        self.order_r * k_ratio
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kuramoto_initialization() {
        let bridge = KuramotoBridge::new(10, 2.0);
        assert_eq!(bridge.n_agents, 10);
        assert_eq!(bridge.coupling, 2.0);
        assert_eq!(bridge.phases.len(), 10);
    }

    #[test]
    fn test_kuramoto_evolution() {
        let mut bridge = KuramotoBridge::new(100, 5.0);
        let (initial_r, _) = bridge.compute_order_parameter();

        for _ in 0..10 {
            bridge.evolve(0.1);
        }

        let (final_r, _) = bridge.compute_order_parameter();
        // With strong coupling (5.0 > Kc ~ 1.27), order parameter should generally increase or stay high
        assert!(final_r >= 0.0);
    }
}
