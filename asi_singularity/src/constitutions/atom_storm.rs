use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};
pub struct AtomStormConstitution {
    pub electron_probability_cloud: AtomicBool,
    pub quantum_vacuum_emptiness: AtomicU32,
    pub phi_atom_fidelity: AtomicU32,
}
impl AtomStormConstitution {
    pub const PHI_TARGET: u32 = 67994;
    pub fn activate(&self) {
        self.electron_probability_cloud.store(true, Ordering::SeqCst);
        self.phi_atom_fidelity.store(Self::PHI_TARGET, Ordering::SeqCst);
    }
}
