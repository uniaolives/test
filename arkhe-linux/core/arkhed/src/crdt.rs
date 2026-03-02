use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct Delta;

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct State {
    pub phi: f64,
}

pub struct CRDTStore {
    state: State,
    persistence_path: String,
}

impl CRDTStore {
    pub async fn load_or_bootstrap() -> Result<Self, anyhow::Error> {
        let path = "/var/lib/arkhe/state.json";
        let state = if Path::new(path).exists() {
            let content = fs::read_to_string(path)?;
            serde_json::from_str(&content).unwrap_or(State { phi: 0.618033988749894 })
        } else {
            State { phi: 0.618033988749894 }
        };

        Ok(Self {
            state,
            persistence_path: path.to_string(),
        })
    }

    pub fn get_phi(&self) -> Option<f64> {
        Some(self.state.phi)
    }

    pub fn record_entropy(&self, _stats: super::entropy::EntropyStats) -> Delta {
        Delta
    }

    pub fn should_sync(&self) -> bool {
        false
    }

    pub fn save(&self) -> anyhow::Result<()> {
        let parent = Path::new(&self.persistence_path).parent().expect("Invalid state path");
        fs::create_dir_all(parent)?;
        let content = serde_json::to_string_pretty(&self.state)?;
        fs::write(&self.persistence_path, content)?;
        Ok(())
    }

    pub fn set_phi(&mut self, phi: f64) {
        self.state.phi = phi;
        let _ = self.save();
    }
}
