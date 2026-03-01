use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct EntropyStats {
    pub cpu_entropy: f64,
    pub memory_entropy: f64,
    pub io_entropy: f64,
    pub global_phi: f64,
}

pub struct EntropyMonitor;

impl EntropyMonitor {
    pub async fn attach_probes() -> Result<Self, anyhow::Error> {
        tracing::info!("Mocking eBPF probes attachment...");
        Ok(Self)
    }

    pub async fn collect(&self) -> EntropyStats {
        EntropyStats {
            cpu_entropy: 0.618,
            memory_entropy: 0.618,
            io_entropy: 0.618,
            global_phi: 0.618033988749894,
        }
    }
}
