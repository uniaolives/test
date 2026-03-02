use std::time::Instant;
use crate::SudoError;

#[derive(Clone, Debug)]
pub struct FragId(pub u32);
impl FragId {
    pub fn to_bytes(&self) -> [u8; 4] { self.0.to_le_bytes() }
    pub fn has_capability(&self, _cap: FragCapability) -> bool { true }
}
pub enum FragCapability { EscalationVerification }

#[derive(Clone, Debug)]
pub struct TMRConsensus {
    pub achieved: bool,
    pub votes_for: usize,
    pub votes_against: usize,
    pub participating_groups: usize,
    pub state_hash: [u8; 32],
}

pub enum ConsensusThreshold { SuperMajority(usize) }
pub struct FragVerification {
    pub frag_id: FragId,
    pub state_hash: [u8; 32],
    pub timestamp: Instant,
}

pub struct CathedralVM;
impl CathedralVM {
    pub async fn execute_tmr_consensus<F>(&self, _f: F, _threshold: ConsensusThreshold) -> Result<TMRConsensus, String>
    where F: Fn(FragId) -> Result<FragVerification, SudoError>
    {
        Ok(TMRConsensus { achieved: true, votes_for: 36, votes_against: 0, participating_groups: 36, state_hash: [0; 32] })
    }
}
