// src/monitoring/immune_system.rs

use std::collections::HashMap;
use crate::monitoring::ghost_vajra_integration::{ThreatLevel};

pub struct ImmuneState {
    pub threat_counts: HashMap<ThreatLevel, u64>,
    pub adaptive_rate: f64,
    pub security_level: SecurityLevel,
}

impl ImmuneState {
    pub fn new() -> Self {
        Self {
            threat_counts: HashMap::new(),
            adaptive_rate: 1.0,
            security_level: SecurityLevel::Normal,
        }
    }

    pub fn record_encounter(&mut self, threat: &ThreatLevel, _penalty: f64, new_phi: f64) {
        *self.threat_counts.entry(threat.clone()).or_insert(0) += 1;

        if threat == &ThreatLevel::Critical || new_phi < 0.70 {
            self.adaptive_rate *= 1.5;
        } else if threat == &ThreatLevel::Noise {
            self.adaptive_rate *= 0.95;
        }
        self.adaptive_rate = self.adaptive_rate.clamp(0.1, 10.0);

        self.security_level = match new_phi {
            phi if phi < 0.70 => SecurityLevel::Maximum,
            phi if phi < 0.75 => SecurityLevel::High,
            phi if phi < 0.85 => SecurityLevel::Elevated,
            _ => SecurityLevel::Normal,
        };
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SecurityLevel {
    Normal, Elevated, High, Maximum,
}
