use std::ffi::c_void;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct C_Handover {
    pub id: u64,
    pub source_id: u64,
    pub target_id: u64,
    pub entropy_cost: f64,
    pub half_life: f64,
}

extern "C" {
    fn arkhe_history_new(hilbert_dim: i32) -> *mut c_void;
    fn arkhe_history_free(ptr: *mut c_void);
    fn arkhe_history_append(ptr: *mut c_void, h: C_Handover);
    fn arkhe_history_von_neumann_entropy(ptr: *mut c_void) -> f64;
    fn arkhe_history_is_bifurcation(ptr: *mut c_void) -> bool;
}

pub struct QuantumHistory {
    ptr: *mut c_void,
}

impl QuantumHistory {
    pub fn new(hilbert_dim: i32) -> Self {
        let ptr = unsafe { arkhe_history_new(hilbert_dim) };
        QuantumHistory { ptr }
    }

    pub fn append_handover(&mut self, handover: C_Handover) {
        unsafe { arkhe_history_append(self.ptr, handover) };
    }

    pub fn entropy(&self) -> f64 {
        unsafe { arkhe_history_von_neumann_entropy(self.ptr) }
    }

    pub fn is_bifurcation(&self) -> bool {
        unsafe { arkhe_history_is_bifurcation(self.ptr) }
    }
}

impl Drop for QuantumHistory {
    fn drop(&mut self) {
        unsafe { arkhe_history_free(self.ptr) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_integration() {
        let mut hist = QuantumHistory::new(2);
        let handover = C_Handover {
            id: 1,
            source_id: 1,
            target_id: 2,
            entropy_cost: 0.3,
            half_life: 1000.0
        };
        hist.append_handover(handover);
        let entropy = hist.entropy();
        println!("Rust FFI entropy: {}", entropy);
        assert!(entropy > 0.0);
    }
}
