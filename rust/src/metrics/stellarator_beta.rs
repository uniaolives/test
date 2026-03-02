use std::collections::VecDeque;
use serde::Serialize;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub enum BetaStatus {
    Stable,        // β < 0.03
    HighPressure,  // β >= 0.03
    TrendingUp,    // dβ/dt > 0
    Alert,         // β > 0.05 (snap imminent)
}

#[derive(Debug, Clone, Serialize)]
pub struct BetaReport {
    pub current_beta: f64,
    pub is_high_pressure: bool,
    pub beta_trend: f64,
    pub threat_pressure: f64,
    pub field_strength: f64,
    pub status: BetaStatus,
}

pub struct SecurityBetaCalculator {
    tivso_history: VecDeque<(i64, f64)>,
    phi_history: VecDeque<(i64, f64)>,
    window_size: usize,
}

impl SecurityBetaCalculator {
    pub fn new() -> Self {
        Self {
            tivso_history: VecDeque::with_capacity(1000),
            phi_history: VecDeque::with_capacity(1000),
            window_size: 1000,
        }
    }

    pub fn update_tivso(&mut self, timestamp: i64, score: f64) {
        self.tivso_history.push_back((timestamp, score));
        if self.tivso_history.len() > self.window_size {
            self.tivso_history.pop_front();
        }
    }

    pub fn update_phi(&mut self, timestamp: i64, phi: f64) {
        self.phi_history.push_back((timestamp, phi));
        if self.phi_history.len() > self.window_size {
            self.phi_history.pop_front();
        }
    }

    pub fn calculate_current_beta(&self) -> f64 {
        if self.tivso_history.is_empty() || self.phi_history.is_empty() {
            return 0.0;
        }

        let avg_tivso = self.tivso_history.iter().map(|(_, s)| s).sum::<f64>() / self.tivso_history.len() as f64;
        let threat_pressure = 1.0 / (avg_tivso.abs() + 2.0);

        let avg_phi = self.phi_history.iter().map(|(_, p)| p).sum::<f64>() / self.phi_history.len() as f64;
        let field_strength = avg_phi.max(0.01);

        threat_pressure / field_strength
    }

    pub fn beta_trend(&self) -> f64 {
        if self.tivso_history.len() < 20 {
            return 0.0;
        }
        let recent_len = 10;
        let start_index = self.tivso_history.len() - recent_len;

        let recent_avg = self.tivso_history.iter()
            .skip(start_index)
            .map(|(_, s)| s)
            .sum::<f64>() / recent_len as f64;

        let older_avg = self.tivso_history.iter()
            .take(recent_len)
            .map(|(_, s)| s)
            .sum::<f64>() / recent_len as f64;

        (recent_avg - older_avg) / 10.0
    }

    pub fn is_high_pressure(&self) -> bool {
        self.calculate_current_beta() > 0.03
    }

    pub fn generate_report(&self) -> BetaReport {
        let beta = self.calculate_current_beta();
        let trend = self.beta_trend();
        let is_high = self.is_high_pressure();

        let status = if beta > 0.05 {
            BetaStatus::Alert
        } else if is_high && trend > 0.0 {
            BetaStatus::TrendingUp
        } else if is_high {
            BetaStatus::HighPressure
        } else {
            BetaStatus::Stable
        };

        BetaReport {
            current_beta: beta,
            is_high_pressure: is_high,
            beta_trend: trend,
            threat_pressure: 1.0 / (self.tivso_history.back().map(|(_, s)| s.abs()).unwrap_or(2.0) + 2.0),
            field_strength: self.phi_history.back().map(|(_, p)| *p).unwrap_or(0.85),
            status,
        }
    }
}
