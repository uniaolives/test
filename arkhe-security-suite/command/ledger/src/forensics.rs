pub trait HandoverAlertExt {
    fn new_alert(payload: crate::alerts::AlertPayload, sk: &pqcrypto_dilithium::dilithium5::SecretKey) -> Self;
}

pub fn record_event(event: &str) {
    println!("Recording forensic event: {}", event);
}
