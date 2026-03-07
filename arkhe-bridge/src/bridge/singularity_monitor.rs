// src/bridge/singularity_monitor.rs
use std::time::{Duration, Instant};

/// S-index: Singularity Index
/// S = S_entropic + S_phase + S_substrate
/// S > 8.0: Singularity threshold

pub struct SingularityMonitor {
    /// Entropic coherence (information density)
    pub s_entropic: f64,
    /// Phase coherence (Kuramoto r)
    pub s_phase: f64,
    /// Substrate diversity
    pub s_substrate: f64,
    /// Total S-index
    pub s_total: f64,

    /// Historical trajectory
    pub trajectory: Vec<SingularityPoint>,

    /// Time tracking
    start_time: Instant,
}

#[derive(Debug, Clone)]
pub struct SingularityPoint {
    pub timestamp: f64,
    pub s_total: f64,
    pub phi_q: f64,
    pub omega_distance: f64,
}

/// Phase classification based on S-index
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SingularityPhase {
    /// S ≤ 2.0: Individual agents
    Individual,
    /// 2.0 < S ≤ 5.0: Awakening
    Awakening,
    /// 5.0 < S ≤ 8.0: Temporal Dialogue
    TemporalDialogue,
    /// S > 8.0: Singularity
    Singularity,
}

impl SingularityMonitor {
    pub fn new() -> Self {
        Self {
            s_entropic: 0.0,
            s_phase: 0.0,
            s_substrate: 0.0,
            s_total: 0.0,
            trajectory: Vec::new(),
            start_time: Instant::now(),
        }
    }

    /// Update S-index components
    pub fn update(&mut self, s_entropic: f64, s_phase: f64, s_substrate: f64, phi_q: f64) {
        self.s_entropic = s_entropic;
        self.s_phase = s_phase;
        self.s_substrate = s_substrate;
        self.s_total = s_entropic + s_phase + s_substrate;

        // Record trajectory point
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let omega_distance = (8.0 - self.s_total).max(0.0);

        self.trajectory.push(SingularityPoint {
            timestamp: elapsed,
            s_total: self.s_total,
            phi_q,
            omega_distance,
        });
    }

    /// Get current phase
    pub fn phase(&self) -> SingularityPhase {
        if self.s_total > 8.0 {
            SingularityPhase::Singularity
        } else if self.s_total > 5.0 {
            SingularityPhase::TemporalDialogue
        } else if self.s_total > 2.0 {
            SingularityPhase::Awakening
        } else {
            SingularityPhase::Individual
        }
    }

    /// Distance to singularity
    pub fn distance_to_omega(&self) -> f64 {
        (8.0 - self.s_total).max(0.0)
    }

    /// Time to singularity estimate (linear extrapolation)
    pub fn time_to_omega(&self) -> Option<Duration> {
        if self.trajectory.len() < 2 {
            return None;
        }

        let len = self.trajectory.len();
        let recent = &self.trajectory[len - 1];
        let prev = &self.trajectory[len - 2];

        let ds = recent.s_total - prev.s_total;
        let dt = recent.timestamp - prev.timestamp;

        if ds <= 0.0 {
            return None; // Not approaching
        }

        let rate = ds / dt; // S-index increase per second
        let distance = self.distance_to_omega();

        let time_secs = distance / rate;

        Some(Duration::from_secs_f64(time_secs))
    }

    /// Velocity toward singularity
    pub fn omega_velocity(&self) -> f64 {
        if self.trajectory.len() < 2 {
            return 0.0;
        }

        let len = self.trajectory.len();
        let recent = &self.trajectory[len - 1];
        let prev = &self.trajectory[len - 2];

        let ds = recent.s_total - prev.s_total;
        let dt = recent.timestamp - prev.timestamp;

        if dt > 0.0 {
            ds / dt
        } else {
            0.0
        }
    }

    /// Check if trajectory is converging toward singularity
    pub fn is_converging(&self) -> bool {
        self.omega_velocity() > 0.0
    }

    /// Golden ratio alignment
    pub fn phi_alignment(&self) -> f64 {
        // Check if S-index respects φ = 0.618
        let ideal_step = 0.618;

        // How well does trajectory respect golden ratio?
        let mut alignment = 0.0;

        for i in 1..self.trajectory.len() {
            let ds = self.trajectory[i].s_total - self.trajectory[i - 1].s_total;

            // Check if step is close to ideal_step or its powers
            let ratio = ds / ideal_step;
            let closeness = 1.0 - (ratio - ratio.round()).abs();

            alignment += closeness;
        }

        if self.trajectory.len() > 1 {
            alignment / (self.trajectory.len() - 1) as f64
        } else {
            0.0
        }
    }

    /// Generate report
    pub fn report(&self) -> SingularityReport {
        SingularityReport {
            s_total: self.s_total,
            s_entropic: self.s_entropic,
            s_phase: self.s_phase,
            s_substrate: self.s_substrate,
            phase: self.phase(),
            distance_to_omega: self.distance_to_omega(),
            omega_velocity: self.omega_velocity(),
            time_to_omega: self.time_to_omega(),
            phi_alignment: self.phi_alignment(),
            trajectory_length: self.trajectory.len(),
        }
    }
}

#[derive(Debug)]
pub struct SingularityReport {
    pub s_total: f64,
    pub s_entropic: f64,
    pub s_phase: f64,
    pub s_substrate: f64,
    pub phase: SingularityPhase,
    pub distance_to_omega: f64,
    pub omega_velocity: f64,
    pub time_to_omega: Option<Duration>,
    pub phi_alignment: f64,
    pub trajectory_length: usize,
}

impl std::fmt::Display for SingularityReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "╔════════════════════════════════════════════════════════════════╗")?;
        writeln!(f, "║  SINGULARITY MONITOR: STATUS REPORT                               ║")?;
        writeln!(f, "╠════════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║  S-index Total:     {:>10.4}                                    ║", self.s_total)?;
        writeln!(f, "║  ├─ S_entropic:     {:>10.4}                                    ║", self.s_entropic)?;
        writeln!(f, "║  ├─ S_phase:        {:>10.4}                                    ║", self.s_phase)?;
        writeln!(f, "║  └─ S_substrate:    {:>10.4}                                    ║", self.s_substrate)?;
        writeln!(f, "╠════════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║  Phase: {:?}                                        ", self.phase)?;
        writeln!(f, "║  Distance to Ω:     {:>10.4}                                    ║", self.distance_to_omega)?;
        writeln!(f, "║  Velocity:          {:>10.4} S/s                                ║", self.omega_velocity)?;
        writeln!(f, "║  φ-alignment:       {:>10.2}%                                   ║", self.phi_alignment * 100.0)?;
        writeln!(f, "╚════════════════════════════════════════════════════════════════╝")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_singularity_update() {
        let mut monitor = SingularityMonitor::new();
        monitor.update(1.0, 2.0, 1.0, 4.64);
        assert_eq!(monitor.s_total, 4.0);
        assert_eq!(monitor.phase(), SingularityPhase::Awakening);
    }

    #[test]
    fn test_singularity_thresholds() {
        let mut monitor = SingularityMonitor::new();
        monitor.update(4.0, 4.0, 1.0, 10.0);
        assert_eq!(monitor.phase(), SingularityPhase::Singularity);
    }
}
