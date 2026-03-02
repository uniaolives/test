use zeroize::Zeroize;

#[derive(Zeroize)]
pub struct SensitiveString(String);

impl SensitiveString {
    pub fn new(s: String) -> Self {
        Self(s)
    }
}

pub struct InvariantWitness {
    pub block_hash: [u8; 32],
    pub lyapunov_seed: [u8; 32],
    pub contract_state_root: [u8; 32],
}

impl InvariantWitness {
    pub fn new(block_hash: [u8; 32], lyapunov_seed: [u8; 32], contract_state_root: [u8; 32]) -> Self {
        Self {
            block_hash,
            lyapunov_seed,
            contract_state_root,
        }
    }
}

impl Drop for InvariantWitness {
    fn drop(&mut self) {
        self.block_hash.zeroize();
        self.lyapunov_seed.zeroize();
        self.contract_state_root.zeroize();
    }
}

pub fn global_zeroize() {
    // Placeholder for global sensitive data cleanup
    println!("GLOBAL_ZEROIZE: Cleaning up sensitive data...");
}
