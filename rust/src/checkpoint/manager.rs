// rust/src/checkpoint/manager.rs
use crate::error::{ResilientError, ResilientResult};
use crate::state::{ResilientState, StateCompressor};
use crate::constitution::CGEValidator;
use crate::wallet::WalletManager;
use crate::checkpoint::{CheckpointResult, CheckpointTrigger, StateScrubber, UploadContext, UploadStrategy};
use crate::network::{ArweaveClient, NostrClient};
use std::time::Instant;
use tokio::sync::RwLock;

pub struct CheckpointManager {
    pub state: RwLock<ResilientState>,
    pub constitution: RwLock<CGEValidator>,
    pub wallet: WalletManager,
    pub arweave: ArweaveClient,
    pub nostr: NostrClient,
    scrubber: StateScrubber,
    #[allow(dead_code)]
    last_checkpoint_time: RwLock<u64>,
    #[allow(dead_code)]
    checkpoint_counter: RwLock<u64>,
}

impl CheckpointManager {
    pub async fn new(
        agent_id: &str,
        wallet: WalletManager,
        arweave: ArweaveClient,
        nostr: NostrClient,
    ) -> ResilientResult<Self> {
        let state = ResilientState::new(agent_id);
        let constitution = CGEValidator::new();
        let scrubber = StateScrubber::default();

        Ok(Self {
            state: RwLock::new(state),
            constitution: RwLock::new(constitution),
            wallet,
            arweave,
            nostr,
            scrubber,
            last_checkpoint_time: RwLock::new(0),
            checkpoint_counter: RwLock::new(0),
        })
    }

    pub async fn checkpoint(&self, trigger: CheckpointTrigger) -> ResilientResult<CheckpointResult> {
        let start_time = Instant::now();
        let mut state = self.state.write().await;

        let previous_tx_id = state.previous_tx_id.clone();
        state.prepare_for_checkpoint(previous_tx_id.clone())?;

        let mut constitution = self.constitution.write().await;
        constitution.validate_state(&state)?;

        let scrubbed_state = self.scrubber.scrub(state.clone())?;
        let compressed = StateCompressor::compress_state(&scrubbed_state)?;
        let compressed_size = compressed.len();

        let upload_context = UploadContext {
            data_size: compressed_size,
            previous_tx_id,
            trigger: trigger.clone(),
        };

        let strategy = UploadStrategy::select(&upload_context, &self.wallet).await?;

        let (tx_id, cost) = match strategy {
            UploadStrategy::TurboFree => {
                let tx_id = self.arweave.upload_via_turbo(&compressed).await?;
                (tx_id, 0)
            }
            UploadStrategy::StandardPaid => {
                let (tx_id, cost) = self.arweave.upload_standard(&compressed).await?;
                (tx_id, cost)
            }
        };

        if state.height == 1 {
            state.genesis_tx_id = tx_id.clone();
        }
        state.previous_tx_id = Some(tx_id.clone());

        self.nostr.announce_checkpoint(&tx_id, &state.agent_id, &format!("{:?}", trigger)).await?;

        let duration = start_time.elapsed();

        Ok(CheckpointResult {
            tx_id,
            size_bytes: compressed_size,
            cost_winston: cost,
            duration_ms: duration.as_millis(),
            strategy_used: strategy,
            validation_passed: true,
            timestamp: timestamp_millis(),
        })
    }
}

fn timestamp_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}
