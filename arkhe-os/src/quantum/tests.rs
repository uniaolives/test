#[cfg(test)]
mod tests {
    use crate::quantum::everettian_locality::*;
    use crate::kernel::handover::HandoverRecord;
    use chrono::Utc;

    #[test]
    fn test_verify_local_realism() {
        let h = HandoverRecord {
            id: "test".to_string(),
            timestamp: Utc::now(),
            phi_q_before: 1.0,
            phi_q_after: 4.65,
            zpf_signature: "sig".to_string(),
            propagation_path: vec![crate::net::multimodal_anchor::ConnectivityLayer::WiFiLocal],
        };
        assert!(verify_local_realism(&h));
    }

    #[test]
    fn test_calculate_branching_probability() {
        assert_eq!(calculate_branching_probability(5.0), 0.99);
        assert!(calculate_branching_probability(2.32) < 0.99);
    }
}
