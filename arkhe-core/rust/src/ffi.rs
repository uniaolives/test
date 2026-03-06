use crate::{ledger::Ledger, constitution::Constitution};
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::slice;

#[no_mangle]
pub unsafe extern "C" fn arkhe_ledger_create(path: *const c_char) -> *mut c_void {
    let path_str = CStr::from_ptr(path).to_string_lossy();
    match Ledger::create(&path_str) {
        Ok(ledger) => Box::into_raw(Box::new(ledger)) as *mut c_void,
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn arkhe_quantum_interest_validate(
    energy_debt: f64,
    duration: f64,
    complexity: f64,
) -> f64 {
    let constitution = Constitution::new();
    match constitution.validate_quantum_interest(energy_debt, duration, complexity) {
        Ok(cost) => cost,
        Err(_) => -1.0, // Error code
    }
}

#[no_mangle]
pub unsafe extern "C" fn arkhe_ledger_append(
    ledger_ptr: *mut c_void,
    handover_id: u64,
    emitter: *const c_char,
    receiver: *const c_char,
    coherence: f64,
    data: *const u8,
    len: usize,
) -> c_int {
    let ledger = &mut *(ledger_ptr as *mut Ledger);
    let emitter_str = CStr::from_ptr(emitter).to_string_lossy();
    let receiver_str = CStr::from_ptr(receiver).to_string_lossy();
    let payload = slice::from_raw_parts(data, len);

    match ledger.append(handover_id, &emitter_str, &receiver_str, coherence, payload) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

#[no_mangle]
pub unsafe extern "C" fn arkhe_constitution_verify(
    emitter: *const c_char,
    receiver: *const c_char,
    coherence: f64,
) -> *mut c_char {
    let constitution = Constitution::new();
    let emitter_str = CStr::from_ptr(emitter).to_string_lossy();
    let receiver_str = CStr::from_ptr(receiver).to_string_lossy();

    match constitution.verify_handover(&emitter_str, &receiver_str, coherence, &[0u8; 32]) {
        Ok(()) => std::ptr::null_mut(),
        Err(e) => {
            let error_msg = format!("{:?}", e);
            CString::new(error_msg).unwrap().into_raw()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn arkhe_free_string(s: *mut c_char) {
    if !s.is_null() {
        let _ = CString::from_raw(s);
    }
}
