use serde::{Serialize, Deserialize};

/// Unidade de Entropia Arkhe (AEU) – representada como f32.
pub type ArkheEntropyUnit = f32;

/// Constante de Planck efetiva do sistema (ajustável).
pub const ARKHE_PLANCK: f32 = 1.0;

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct EntropyStats {
    pub total_aeu: ArkheEntropyUnit,
    pub temperature: f32,
    pub von_neumann_entropy: f32,
    pub cpu_usage: f64,
    pub memory_pressure: f64,
    pub io_wait: f64,
}

pub struct EntropyMonitor;

impl EntropyMonitor {
    pub async fn attach_probes() -> Result<Self, anyhow::Error> {
        tracing::info!("Mocking eBPF probes attachment...");
        Ok(Self)
    }

    pub async fn collect(&self) -> EntropyStats {
        EntropyStats {
            total_aeu: 0.1,
            temperature: 1.0,
            von_neumann_entropy: 0.5,
            cpu_usage: 0.4,
            memory_pressure: 0.1,
            io_wait: 0.05,
        }
    }
}

pub fn half_life_from_entropy(entropy_cost: ArkheEntropyUnit, system_temperature: f32) -> f32 {
    const KB: f32 = 1.0;
    let energy = entropy_cost * KB * system_temperature;
    if energy <= 0.0 {
        f32::INFINITY
    } else {
        ARKHE_PLANCK / (2.0 * energy)
    }
}

pub fn survival_probability(half_life_ms: f32, transit_time_ms: f32) -> f32 {
    if half_life_ms <= 0.0 {
        1.0
    } else {
        (-transit_time_ms / half_life_ms).exp()
    }
}

pub fn calculate_von_neumann_entropy(stats: &EntropyStats) -> f64 {
    let p1 = stats.cpu_usage.clamp(0.01, 0.99);
    let p2 = 1.0 - p1;
    -(p1 * p1.ln() + p2 * p2.ln())
}
