// rust/src/astrophysics/kardashev.rs

pub enum KardashevType {
    TypeI,
    TypeII,
    TypeIII,
}

pub struct CivilizationalMetrics {
    pub level: f64,
    pub energy_usage_watts: f64,
    pub compute_ops_per_sec: f64,
}

impl CivilizationalMetrics {
    pub fn current() -> Self {
        Self {
            level: 2.0,
            energy_usage_watts: 3.828e26,
            compute_ops_per_sec: 1e42,
        }
    }

    pub fn get_type(&self) -> KardashevType {
        if self.level >= 3.0 {
            KardashevType::TypeIII
        } else if self.level >= 2.0 {
            KardashevType::TypeII
        } else {
            KardashevType::TypeI
        }
    }
}
