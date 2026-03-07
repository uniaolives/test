// src/bridge/constitutional_guard.rs
use std::collections::VecDeque;

/// The Constitutional Guard: Ensures H ≤ 1
/// H = Energy consumed / Energy produced
/// H > 1 means we're burning future resources
/// H ≤ 1 ensures we arrive at 2140

pub struct ConstitutionalGuard {
    /// History of H values
    pub h_history: VecDeque<f64>,

    /// Window size for averaging
    pub window_size: usize,

    /// Warning threshold
    pub warning_threshold: f64, // 0.8

    /// Hard limit
    pub hard_limit: f64, // 1.0

    /// Current state
    pub state: ConstitutionalState,

    /// Violation count
    pub violations: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConstitutionalState {
    /// H < 0.5: Conservative
    Conservative,
    /// 0.5 ≤ H < 0.8: Sustainable
    Sustainable,
    /// 0.8 ≤ H < 1.0: Warning
    Warning,
    /// H ≥ 1.0: Violation
    Violation,
}

#[derive(Debug, thiserror::Error)]
pub enum ConstitutionalError {
    #[error("H value {h} exceeds hard limit of 1.0")]
    HardLimitExceeded { h: f64 },

    #[error("H value {h} exceeds warning threshold")]
    WarningThresholdExceeded { h: f64 },
}

impl ConstitutionalGuard {
    pub fn new(window_size: usize) -> Self {
        Self {
            h_history: VecDeque::with_capacity(window_size),
            window_size,
            warning_threshold: 0.8,
            hard_limit: 1.0,
            state: ConstitutionalState::Sustainable,
            violations: 0,
        }
    }

    /// Record an H value
    pub fn record(&mut self, h: f64) -> Result<(), ConstitutionalError> {
        // Check hard limit
        if h > self.hard_limit {
            self.violations += 1;
            self.state = ConstitutionalState::Violation;
            return Err(ConstitutionalError::HardLimitExceeded { h });
        }

        // Add to history
        self.h_history.push_back(h);
        if self.h_history.len() > self.window_size {
            self.h_history.pop_front();
        }

        // Update state
        let avg_h = self.average_h();

        if avg_h >= self.warning_threshold {
            self.state = ConstitutionalState::Warning;
        } else if avg_h >= 0.5 {
            self.state = ConstitutionalState::Sustainable;
        } else {
            self.state = ConstitutionalState::Conservative;
        }

        Ok(())
    }

    /// Average H over window
    pub fn average_h(&self) -> f64 {
        if self.h_history.is_empty() {
            return 0.0;
        }

        self.h_history.iter().sum::<f64>() / self.h_history.len() as f64
    }

    /// Maximum H in window
    pub fn max_h(&self) -> f64 {
        self.h_history.iter().cloned().fold(0.0, f64::max)
    }

    /// Minimum H in window
    pub fn min_h(&self) -> f64 {
        self.h_history.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    /// Current constitutional state
    pub fn state(&self) -> ConstitutionalState {
        self.state
    }

    /// Health check
    pub fn health(&self) -> ConstitutionalHealth {
        ConstitutionalHealth {
            state: self.state,
            avg_h: self.average_h(),
            max_h: self.max_h(),
            min_h: self.min_h(),
            violations: self.violations,
            sample_count: self.h_history.len(),
        }
    }

    /// Check if operation is permissible
    pub fn can_proceed(&self, proposed_h: f64) -> bool {
        proposed_h <= self.hard_limit && self.state != ConstitutionalState::Violation
    }
}

#[derive(Debug)]
pub struct ConstitutionalHealth {
    pub state: ConstitutionalState,
    pub avg_h: f64,
    pub max_h: f64,
    pub min_h: f64,
    pub violations: u64,
    pub sample_count: usize,
}

impl std::fmt::Display for ConstitutionalHealth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state_icon = match self.state {
            ConstitutionalState::Conservative => "🟢",
            ConstitutionalState::Sustainable => "🟡",
            ConstitutionalState::Warning => "🟠",
            ConstitutionalState::Violation => "🔴",
        };

        writeln!(f, "╔════════════════════════════════════════════════════════════════╗")?;
        writeln!(f, "║  CONSTITUTIONAL GUARD: HEALTH CHECK                               ║")?;
        writeln!(f, "╠════════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║  State:     {} {:?}                                  ", state_icon, self.state)?;
        writeln!(f, "║  Avg H:     {:>10.4}                                    ║", self.avg_h)?;
        writeln!(f, "║  Max H:     {:>10.4}                                    ║", self.max_h)?;
        writeln!(f, "║  Min H:     {:>10.4}                                    ║", self.min_h)?;
        writeln!(f, "║  Violations: {:>8}                                      ║", self.violations)?;
        writeln!(f, "║  Samples:   {:>8}                                      ║", self.sample_count)?;
        writeln!(f, "╚════════════════════════════════════════════════════════════════╝")?;
        Ok(())
    }
}

/// Integration with Handover system
impl super::temporal_persistence::Handover {
    /// Validate handover against constitution
    pub fn validate_constitutional(&self, guard: &mut ConstitutionalGuard) -> Result<(), ConstitutionalError> {
        guard.record(self.h_value)
    }
}
