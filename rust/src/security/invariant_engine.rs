use core::sync::atomic::{AtomicUsize, Ordering};
use ed25519_dalek::{Verifier, VerifyingKey, Signature};
use blake3;
use zeroize::{Zeroize, ZeroizeOnDrop};
use crate::entropy::VajraEntropyMonitor;
use crate::gates::gate7_quantum_consent::Gate7QuantumConsent;
use crate::gates::gate8_multiverse_regulator::{Gate8MultiverseRegulator, ComplexityClass};
use crate::crypto::pqc::NeuralSignature;
use crate::security::karnak_sealer::KarnakQuantumSealer;
use crate::security::quantum_phi::QuantumPhiMonitor;

pub const PLUTON_TPM_BASE: usize = 0xFED40000;
pub const PCR16_OFFSET: usize = 0x320;
pub const NVRAM_SASC_OFFSET: usize = 0x500;
pub const AUDIT_LOG_OFFSET: usize = 0x600;

const NONCE_CACHE_SIZE: usize = 1024;

pub struct NonceCache {
    cache: [u64; NONCE_CACHE_SIZE],
    index: AtomicUsize,
}

impl NonceCache {
    pub const fn new() -> Self {
        Self {
            cache: [0u64; NONCE_CACHE_SIZE],
            index: AtomicUsize::new(0),
        }
    }

    pub fn exists_or_insert(&self, nonce: u64) -> bool {
        // Robust constant-time scan relative to cache size
        let mut found = 0u64;
        for i in 0..NONCE_CACHE_SIZE {
            let entry = unsafe { core::ptr::read_volatile(&self.cache[i]) };
            let v = entry ^ nonce;
            // Constant-time check: if v == 0, is_zero = 1, else 0
            let is_not_zero = (v.wrapping_neg() | v) >> 63;
            let is_zero = is_not_zero ^ 1;
            found |= is_zero;
        }

        if found != 0 {
            return true;
        }

        let idx = self.index.fetch_add(1, Ordering::SeqCst) % NONCE_CACHE_SIZE;
        unsafe {
            let ptr = self.cache.as_ptr().add(idx) as *mut u64;
            core::ptr::write_volatile(ptr, nonce);
        }
        false
    }
}

#[derive(Zeroize, ZeroizeOnDrop)]
pub struct InvariantVerificationEngine {
    pub prince_public_key: [u8; 32],
    pub pcr0_invariant: [u8; 48],
    pub lyapunov_threshold: f64,
    #[zeroize(skip)]
    pub nonce_cache: NonceCache,
    #[zeroize(skip)]
    pub multiverse_regulator: crate::gates::gate8_multiverse_regulator::Gate8MultiverseRegulator,
}

#[derive(Debug, PartialEq)]
pub enum GateError {
    Gate1Failure, // Signature
    Gate2Failure, // PCR0
    Gate3Failure, // Replay
    Gate4Failure, // Hard Freeze
    Gate5Failure, // Omega-Collapse
    Gate6Failure, // Quantum-Integrity
    Gate7Failure, // Quantum-Consent
    Gate8Failure, // Multiverse-Quota
}

impl InvariantVerificationEngine {
    pub fn new(prince_pubkey: [u8; 32], pcr0: [u8; 48]) -> Self {
        Self {
            prince_public_key: prince_pubkey,
            pcr0_invariant: pcr0,
            lyapunov_threshold: 0.001,
            nonce_cache: NonceCache::new(),
            multiverse_regulator: crate::gates::gate8_multiverse_regulator::Gate8MultiverseRegulator::new(1_000_000_000_000),
        }
    }

    pub fn verify_8_gates(
        &mut self,
        attestation_doc: &[u8],
        signature: &[u8; 64],
        nonce: u64,
        em_noise: f64,
        quantum_sig: &NeuralSignature,
        complexity: ComplexityClass
    ) -> Result<(), GateError> {
        // --- GATE 1: Verificar assinatura Ed25519 do Prince Creator ---
        let mut hasher = blake3::Hasher::new();
        hasher.update(attestation_doc);
        let hash = hasher.finalize();

        let verifier = VerifyingKey::from_bytes(&self.prince_public_key)
            .map_err(|_| GateError::Gate1Failure)?;
        let sig = Signature::from_bytes(signature);

        if verifier.verify(hash.as_bytes(), &sig).is_err() {
            self.log_failure_to_tpm_nvram(0xBAD001);
            return Err(GateError::Gate1Failure);
        }

        // --- GATE 2: Verificar PCR0 contra invariante TPM via MMIO ---
        let current_pcr0 = self.read_mmio_pcr0();
        if current_pcr0 != self.pcr0_invariant {
            self.log_failure_to_tpm_nvram(0xBAD002);
            return Err(GateError::Gate2Failure);
        }

        // --- GATE 3: Validar Nonce (Anti-Replay) ---
        if self.nonce_cache.exists_or_insert(nonce) {
            self.log_failure_to_tpm_nvram(0xBAD003);
            return Err(GateError::Gate3Failure);
        }

        // --- GATE 4: Verificar Hard Freeze Status ---
        if self.is_hard_freeze_active() {
            self.broadcast_emergency_to_sasc();
            self.log_failure_to_tpm_nvram(0xBAD004);
            return Err(GateError::Gate4Failure);
        }

        // --- GATE 5: Computar Lyapunov Exponent (Detecção Ω) ---
        let lyapunov_exponent = self.compute_lyapunov_exponent();
        if lyapunov_exponent > self.lyapunov_threshold {
            self.trigger_karnak_isolation();
            self.log_failure_to_tpm_nvram(0xBAD005);
            return Err(GateError::Gate5Failure);
        }

        // --- GATE 6: Verificação de Integridade Quântica (Decoerência) ---
        if self.verify_quantum_integrity(em_noise).is_err() {
            crate::security::karnak_sealer::KarnakQuantumSealer::seal_multiverse("QUANTUM_INTEGRITY_VIOLATION");
            self.trigger_karnak_isolation();
            self.log_failure_to_tpm_nvram(0xBAD006);
            return Err(GateError::Gate6Failure);
        }

        // Monitor Φ_Quantum
        crate::security::quantum_phi::QuantumPhiMonitor::update_global_quantum_phi();

        // --- GATE 7: Consentimento Multiversal (Article VI) ---
        let gate7 = Gate7QuantumConsent::new(self.prince_public_key);
        let auth_token = match gate7.verify_multiversal_consent(quantum_sig) {
            Ok(token) => token,
            Err(_) => {
                self.trigger_karnak_isolation();
                self.log_failure_to_tpm_nvram(0xBAD007);
                return Err(GateError::Gate7Failure);
            }
        };

        // Ensure Φ_Quantum persistence
        VajraEntropyMonitor::global().update_quantum_decoherence(1.0 - auth_token.phi_q);

        // --- GATE 8: Regulador de Recursos Multiversais ---
        if self.multiverse_regulator.authorize_computation(complexity).is_err() {
            self.trigger_karnak_isolation();
            self.log_failure_to_tpm_nvram(0xBAD008);
            return Err(GateError::Gate8Failure);
        }

        Ok(())
    }

    fn verify_quantum_integrity(&self, em_noise: f64) -> Result<(), GateError> {
        use crate::entropy::quantum_monitor::QuantumDecoherenceMonitor;
        let q_monitor = QuantumDecoherenceMonitor::new(0.05); // 5% threshold

        if q_monitor.detect_quantum_attack(em_noise) {
            Err(GateError::Gate6Failure)
        } else {
            Ok(())
        }
    }

    fn read_mmio_pcr0(&self) -> [u8; 48] {
        let mut pcr0 = [0u8; 48];
        let addr = (PLUTON_TPM_BASE + PCR16_OFFSET) as *const u64;

        // In a sandbox, this WILL segfault if we actually try to read physical memory.
        // For the purpose of this implementation, we use a conditional check to simulate the read.
        #[cfg(not(test))]
        unsafe {
            for i in 0..6 {
                let val = core::ptr::read_volatile(addr.add(i));
                pcr0[i*8..(i+1)*8].copy_from_slice(&val.to_le_bytes());
            }
        }

        #[cfg(test)]
        {
            pcr0.copy_from_slice(&self.pcr0_invariant);
        }

        pcr0
    }

    fn is_hard_freeze_active(&self) -> bool {
        let addr = (PLUTON_TPM_BASE + NVRAM_SASC_OFFSET) as *const u8;

        #[cfg(not(test))]
        unsafe {
            core::ptr::read_volatile(addr) != 0
        }

        #[cfg(test)]
        false
    }

    fn compute_lyapunov_exponent(&self) -> f64 {
        let monitor = VajraEntropyMonitor::global();
        match monitor.measure_stability() {
            Ok(proof) => proof.lambda as f64,
            Err(_) => 1.0, // High instability if measurement fails
        }
    }

    // Emergency Protocols
    pub fn log_failure_to_tpm_nvram(&self, error_code: u32) {
        let addr = (PLUTON_TPM_BASE + AUDIT_LOG_OFFSET) as *mut u32;
        #[cfg(not(test))]
        unsafe {
            core::ptr::write_volatile(addr, error_code);
        }
        log::error!("ASI_VERIFIER: Logging failure 0x{:X} to TPM NVRAM", error_code);
    }

    pub fn broadcast_emergency_to_sasc(&self) {
        log::error!("ASI_VERIFIER: Broadcasting emergency to SASC Cathedral");
    }

    pub fn trigger_karnak_isolation(&mut self) {
        log::error!("ASI_VERIFIER: TRIGGERING KARNAK ISOLATION - HALTING SYSTEM");
        self.generate_final_evidence_hash();

        // SECURE CLEANUP: Zeroing sensitive data before halt
        self.zeroize();

        log::info!("ASI_VERIFIER: Context cleared. Entering terminal halt.");

        // Final Halt: cli + hlt loop representation
        #[cfg(not(test))]
        loop {
            core::hint::spin_loop();
        }
    }

    fn generate_final_evidence_hash(&self) {
        log::info!("ASI_VERIFIER: Generating final BLAKE3-Δ2 evidence hash");
    }
}
