#[derive(Clone)]
pub struct EntropyStats;

pub struct EntropyMonitor;

impl EntropyMonitor {
    pub async fn attach_probes() -> Result<Self, anyhow::Error> {
        Ok(Self)
    }

    pub async fn collect(&self) -> EntropyStats {
        EntropyStats
    }
}
