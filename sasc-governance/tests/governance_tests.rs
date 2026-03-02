use sasc_governance::Cathedral;
use sasc_governance::types::{CloudDomain, Decision, DecisionSignature};

#[test]
fn test_submit_global_decision() {
    let cathedral = Cathedral::instance();
    let decision = Decision {
        id: sasc_governance::types::DecisionId([0; 32]),
        agent_id: "agent_001".to_string(),
        content: "Propose civilizational initiation".to_string(),
        signature: DecisionSignature {
            prince_veto: false,
            signature_bytes: vec![0u8; 64],
        },
        action_hash: [0u8; 32],
        is_critical: false,
        affects_rights: false,
        human_approval: None,
        decision_time: 0,
        explanation: None,
        perspective_count: 5,
    };

    let result = cathedral.submit_global_decision(decision, CloudDomain::WindowsServerGov);
    assert!(result.is_ok());
    let decision_id = result.unwrap();
    println!("Decision ID: {:?}", decision_id);
}

#[test]
fn test_prince_veto() {
    let cathedral = Cathedral::instance();
    let decision = Decision {
        id: sasc_governance::types::DecisionId([0; 32]),
        agent_id: "agent_001".to_string(),
        content: "Dangerous proposal".to_string(),
        signature: DecisionSignature {
            prince_veto: true,
            signature_bytes: vec![0u8; 64],
        },
        action_hash: [0u8; 32],
        is_critical: false,
        affects_rights: false,
        human_approval: None,
        decision_time: 0,
        explanation: None,
        perspective_count: 5,
    };

    let result = cathedral.submit_global_decision(decision, CloudDomain::AwsNitroGovCloud);
    assert!(result.is_err());
    // Verify hard freeze is active
    let gov = cathedral.governance.lock().unwrap();
    assert!(gov.hard_freeze_status);
}

#[test]
fn test_sot_mandate_violation() {
    let cathedral = Cathedral::instance();
    // Reset hard freeze status for test
    {
        let mut gov = cathedral.governance.lock().unwrap();
        gov.hard_freeze_status = false;
    }

    let decision = Decision {
        id: sasc_governance::types::DecisionId([0; 32]),
        agent_id: "monolithic_agent".to_string(),
        content: "High phi decision".to_string(),
        signature: DecisionSignature {
            prince_veto: false,
            signature_bytes: vec![0u8; 64],
        },
        action_hash: [0; 32],
        is_critical: false,
        affects_rights: false,
        human_approval: None,
        decision_time: 0,
        explanation: None,
        perspective_count: 2, // < 3 required for phi > 0.70
    };

    // Cathedral::compute_noosphere_coherence returns 0.75 mock
    let result = cathedral.submit_global_decision(decision, CloudDomain::CloudflareQuantum);

    match result {
        Err(sasc_governance::types::HardFreeze::Triggered(reason)) => {
            assert!(reason.contains("SOT_MANDATE_VIOLATION"));
        },
        _ => panic!("Should have failed with SOT_MANDATE_VIOLATION"),
    }
}
