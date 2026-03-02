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
    };

    let result = cathedral.submit_global_decision(decision, CloudDomain::AwsNitroGovCloud);
    assert!(result.is_err());
    // Verify hard freeze is active
    let gov = cathedral.governance.lock().unwrap();
    assert!(gov.hard_freeze_status);
}
