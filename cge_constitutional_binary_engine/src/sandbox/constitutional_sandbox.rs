// src/sandbox/constitutional_sandbox.rs
use crate::BinaryAnalysis;
use crate::BinaryEngineConfig;
use crate::BinaryError;
use serde::{Serialize, Deserialize};

pub struct Sandbox { pub id: u64 }
impl Sandbox { pub fn id(&self) -> u64 { self.id } }

pub struct SandboxFactory;
impl SandboxFactory {
    pub fn new(_config: &BinaryEngineConfig) -> Result<Self, BinaryError> { Ok(Self) }
    pub async fn create_sandbox(&self, _analysis: &BinaryAnalysis) -> Result<Sandbox, BinaryError> {
        Ok(Sandbox { id: 1 })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SandboxPolicy { Strict }

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResourceLimits;

impl ResourceLimits {
    pub fn default() -> Self { Self }
}
