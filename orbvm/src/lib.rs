pub mod error;
pub mod orb;
pub mod memory;
pub mod temporal;
pub mod network;
pub mod industrial;
pub mod crypto;
pub mod blockchain;
pub mod ssh;
pub mod agi_loop;
pub mod runtime;

pub mod prelude {
    pub use crate::orb::{OrbPayload, OrbVM};
    pub use crate::OrbVMConfig;
}

pub const PI_DAY_2026: i64 = 1742241655;
pub const TARGET_2140: i64 = 4584533760;
pub const GOLDEN_RATIO: f64 = 1.61803398875;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct OrbVMConfig {
    pub n_oscillators: usize,
}

pub type Result<T> = std::result::Result<T, anyhow::Error>;

pub fn init(config: OrbVMConfig) -> Result<orb::OrbVM> {
    Ok(orb::OrbVM::new(config))
}
