use std::time::SystemTime;

pub struct SimpleThermalizer {
    phi: f64,
    history: Vec<(SystemTime, f64)>,
}

impl SimpleThermalizer {
    pub fn new() -> Self {
        Self {
            phi: 0.618,
            history: Vec::new(),
        }
    }

    pub fn set_phi(&mut self, val: f64) {
        self.phi = val.clamp(0.0, 1.0);
        self.history.push((SystemTime::now(), self.phi));
        if self.history.len() > 2000 {
            self.history.remove(0);
        }
    }

    pub fn current_phi(&mut self) -> f64 {
        // Simulates a drift towards 0.618
        let target = 0.618033988749894;
        let drift_rate = 0.001;

        if (self.phi - target).abs() > 0.0001 {
            if self.phi < target {
                self.phi = (self.phi + drift_rate).min(target);
            } else {
                self.phi = (self.phi - drift_rate).max(target);
            }
        }

        self.phi = self.phi.clamp(0.0, 1.0);
        self.history.push((SystemTime::now(), self.phi));
        if self.history.len() > 2000 {
            self.history.remove(0);
        }
        self.phi
    }

    pub fn regime(&self) -> &'static str {
        if self.phi < 0.25 {
            "crystalline"
        } else if (self.phi - 0.618).abs() < 0.05 {
            "critical (golden)"
        } else if self.phi > 0.75 {
            "plasma"
        } else {
            "balanced"
        }
    }

    pub fn get_history(&self, n: usize) -> Vec<(SystemTime, f64)> {
        self.history.iter().rev().take(n).cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi_drift_to_critical() {
        let mut therm = SimpleThermalizer::new();
        therm.set_phi(0.1); // Começa cristalino

        // Simula 1000 atualizações (1000 segundos)
        for _ in 0..1000 {
            let _ = therm.current_phi();
        }

        let final_phi = therm.current_phi();
        // Deve ter driftado para próximo de 0.618
        assert!(final_phi > 0.5, "Deve driftar para cima de 0.5, got {}", final_phi);
        assert!(final_phi < 0.9, "Não deve passar de 0.9, got {}", final_phi);
    }

    #[test]
    fn test_phi_regime_classification() {
        let mut therm = SimpleThermalizer::new();

        therm.set_phi(0.1);
        assert_eq!(therm.regime(), "crystalline");

        therm.set_phi(0.618);
        assert_eq!(therm.regime(), "critical (golden)");

        therm.set_phi(0.9);
        assert_eq!(therm.regime(), "plasma");
    }

    #[test]
    fn test_phi_clamping() {
        let mut therm = SimpleThermalizer::new();

        therm.set_phi(-0.5);
        // drift takes it to 0.001 if target is 0.618
        let phi = therm.current_phi();
        assert!(phi >= 0.0 && phi <= 0.001);

        therm.set_phi(1.5);
        let phi = therm.current_phi();
        assert!(phi <= 1.0 && phi >= 0.999);
    }

    #[test]
    fn test_history_retention() {
        let mut therm = SimpleThermalizer::new();

        for i in 0..1500 {
            therm.set_phi(0.1 + (i as f64 * 0.0005));
        }

        let history = therm.get_history(100);
        assert_eq!(history.len(), 100);
        // Último valor deve ser próximo de 0.85
        assert!(history[0].1 > 0.8);
    }
}
