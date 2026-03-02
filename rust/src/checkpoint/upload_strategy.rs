// rust/src/checkpoint/upload_strategy.rs
use crate::error::ResilientResult;
use crate::wallet::{WalletManager, Network};
use crate::checkpoint::CheckpointTrigger;

#[derive(Debug, Clone, Copy)]
pub enum UploadStrategy {
    TurboFree,
    StandardPaid,
}

pub struct UploadContext {
    pub data_size: usize,
    pub previous_tx_id: Option<String>,
    pub trigger: CheckpointTrigger,
}

impl UploadStrategy {
    pub async fn select(context: &UploadContext, wallet: &WalletManager) -> ResilientResult<Self> {
        if context.data_size <= 102_400 && wallet.network() == Network::Turbo {
            Ok(UploadStrategy::TurboFree)
        } else {
            Ok(UploadStrategy::StandardPaid)
        }
    }
}
