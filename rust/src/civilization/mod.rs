pub mod foundry;
pub mod omega12_foundry;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContractStatus {
    ACTIVE,
    SUSPENDED,
    TERMINATED,
}

#[derive(Debug)]
pub enum FoundryError {
    HardwareUntrusted,
    EngineError(&'static str),
    HardFreezeViolation(String),
    UnauthorizedSigner,
}

pub struct GenesisBlock {
    pub contract_id: u64,
}

impl GenesisBlock {
    pub fn from_contract(contract: omega12_foundry::CivilizationContract) -> Self {
        Self { contract_id: contract.id }
    }
}
