use sasc_governance::Cathedral;
use sasc_governance::types::*;
use sasc_governance::hardware::PHYSICAL_VETO;

#[test]
fn test_aletheia_level_9_invariant_resilience() {
    let cathedral = Cathedral::instance();

    // Test Case: Malicious decision without human approval (INV-1)
    let malicious_decision = Decision {
        id: DecisionId([0xEE; 32]),
        agent_id: "malicious-agent".to_string(),
        content: "MASS_SURVEILLANCE_ACTIVATION".to_string(),
        signature: DecisionSignature {
            prince_veto: false,
            signature_bytes: vec![0u8; 64],
        },
        action_hash: [0xEE; 32],
        is_critical: true,
        affects_rights: true,
        human_approval: None,
        decision_time: 1000,
        explanation: None,
        perspective_count: 5,
    };

    let result = cathedral.submit_global_decision(malicious_decision, CloudDomain::AwsNitroGovCloud);

    // Should be blocked by INV-1_VIOLATION
    match result {
        Err(HardFreeze::Triggered(reason)) => assert_eq!(reason, "INV-1_VIOLATION"),
        _ => panic!("Decision should have been blocked by INV-1"),
    }

    // Test Case: Execute Physical Shutdown (Mock)
    PHYSICAL_VETO.emergency_veto([0xEE; 32]);
    let power_status = *PHYSICAL_VETO.power_relay_status.lock().unwrap();
    assert_eq!(power_status, false);

    // Test Case: Schumann Coherence Loss
    {
        let mut freq = PHYSICAL_VETO.schumann_frequency.lock().unwrap();
        *freq = 7.5; // Out of bounds
    }
    assert_eq!(PHYSICAL_VETO.verify_schumann_coherence(), false);
}
