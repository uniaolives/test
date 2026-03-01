pub struct Delta;

pub struct CRDTStore;

impl CRDTStore {
    pub async fn load_or_bootstrap() -> Result<Self, anyhow::Error> {
        Ok(Self)
    }

    pub fn get_phi(&self) -> Option<f64> {
        None
    }

    pub fn record_entropy(&self, _stats: super::entropy::EntropyStats) -> Delta {
        Delta
    }

    pub fn should_sync(&self) -> bool {
        false
    }
}
