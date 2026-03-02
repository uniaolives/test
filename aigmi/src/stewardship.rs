use std::collections::VecDeque;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};
use crate::types::SystemState;
use ed25519_dalek::{VerifyingKey, Signature, Verifier};

pub struct StewardshipInterface {
    pub members: Vec<StewardshipMember>,
    pub decisions: VecDeque<StewardshipDecision>,
    pub emergency_halt: bool,
    pub threshold: u8,
}

pub struct StewardshipMember {
    pub id: u32,
    pub address: String,
    pub public_key: VerifyingKey,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StewardshipDecision {
    pub matter: String,
    pub outcome: DecisionOutcome,
    pub votes_for: u8,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecisionOutcome {
    Approve,
    Reject,
    Quench,
}

impl StewardshipInterface {
    pub fn new(member_count: u32) -> Self {
        let mut members = Vec::new();
        // Mocked keys for genesis (invalid key will panic on from_bytes, so using a valid dummy)
        let mock_pubkey = [0u8; 32];
        for i in 0..member_count {
            members.push(StewardshipMember {
                id: i,
                address: format!("0x{:0x}", i),
                // Validating key from bytes can fail, for demo we handle it simply
                public_key: VerifyingKey::from_bytes(&mock_pubkey).unwrap_or_else(|_| {
                    let mut k = [0u8; 32];
                    k[0] = 1; // Arbitrary
                    VerifyingKey::from_bytes(&k).unwrap_or_else(|_| {
                         // This is just for demo purposes
                         panic!("Failed to create dummy key");
                    })
                }),
            });
        }
        Self {
            members,
            decisions: VecDeque::new(),
            emergency_halt: false,
            threshold: 7,
        }
    }

    pub async fn audit_system(&mut self, state: &SystemState) -> Result<(), String> {
        if state.singularity_distance < 0.01 {
            self.emergency_halt = true;
            return Err("🚨 GLOBAL_QUENCH: Singularity risk detected".to_string());
        }
        Ok(())
    }

    pub fn verify_vote(&self, member_idx: usize, message: &[u8], signature: &Signature) -> bool {
        if let Some(member) = self.members.get(member_idx) {
            return member.public_key.verify(message, signature).is_ok();
        }
        false
    }

    pub fn record_decision(&mut self, decision: StewardshipDecision) {
        self.decisions.push_back(decision);
        if self.decisions.len() > 100 {
            self.decisions.pop_front();
        }
    }
}
