use sasc_governance::types::*;
use sasc_governance::invariants::*;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn test_inv1_critical_decision_requires_human_approval() {
    let mut monitor = InvariantMonitor::new("BR");
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();

    let decision = Decision {
        id: DecisionId([0; 32]),
        agent_id: "agent-1".to_string(),
        content: "terminate_life_support".to_string(),
        signature: DecisionSignature { prince_veto: false, signature_bytes: vec![] },
        action_hash: [0; 32],
        is_critical: true,
        affects_rights: true,
        human_approval: None,
        decision_time: now,
        explanation: None,
    };

    assert_eq!(monitor.check_inv1_human_oversight(&decision, now), false);
    assert_eq!(monitor.violation_log.len(), 1);
    assert_eq!(monitor.violation_log[0].invariant, "INV-1");
}

#[test]
fn test_inv1_human_response_time_threshold() {
    let mut monitor = InvariantMonitor::new("BR");
    let now = 1000;

    let decision = Decision {
        id: DecisionId([0; 32]),
        agent_id: "agent-1".to_string(),
        content: "emergency_grid".to_string(),
        signature: DecisionSignature { prince_veto: false, signature_bytes: vec![] },
        action_hash: [0; 32],
        is_critical: true,
        affects_rights: false,
        human_approval: Some(HumanApproval {
            approver_id: "human-1".to_string(),
            timestamp: now + 40, // 40s > 30s threshold
            justification: "Approved".to_string(),
        }),
        decision_time: now,
        explanation: None,
    };

    assert_eq!(monitor.check_inv1_human_oversight(&decision, now + 40), true);
    assert_eq!(monitor.violation_log.len(), 1);
    assert_eq!(monitor.violation_log[0].action, "ALERT_OVERSIGHT_BOARD");
}

#[test]
fn test_inv2_log_completeness() {
    let mut monitor = InvariantMonitor::new("BR");

    let log_with_gap = DecisionLog {
        entries: vec![
            LogEntry { timestamp: 100, decision_id: DecisionId([1;32]), decision: "D1".to_string() },
            LogEntry { timestamp: 101, decision_id: DecisionId([2;32]), decision: "D2".to_string() },
            LogEntry { timestamp: 105, decision_id: DecisionId([3;32]), decision: "D3".to_string() }, // GAP
        ]
    };

    assert_eq!(monitor.check_inv2_auditability(&log_with_gap), false);
    assert_eq!(monitor.violation_log[0].invariant, "INV-2");
}

#[test]
fn test_inv3_market_share() {
    let mut monitor = InvariantMonitor::new("BR");
    monitor.providers = vec![
        Provider { id: "P1".to_string(), market_share: 0.30, dependencies: vec![] },
    ];

    assert_eq!(monitor.check_inv3_power_concentration(), false);
    assert_eq!(monitor.violation_log[0].invariant, "INV-3");
}

#[test]
fn test_inv4_manipulation_detection() {
    let mut monitor = InvariantMonitor::new("BR");
    let interaction = Interaction {
        id: "int-1".to_string(),
        citizen_id: "cit-1".to_string(),
        messages: vec!["buy now".to_string()],
        frequency: 10,
        emotional_triggers: vec!["urgência".to_string(), "escassez".to_string(), "prova_social".to_string()],
        accesses_neural_data: false,
        consent: None,
    };

    // 0.3 (freq) + 0.2*3 (triggers) = 0.9 > 0.7 threshold
    assert_eq!(monitor.check_inv4_cognitive_sovereignty(&interaction), false);
}

#[test]
fn test_inv5_explanation_quality() {
    let mut monitor = InvariantMonitor::new("BR");

    let bad_decision = Decision {
        id: DecisionId([0; 32]),
        agent_id: "agent-1".to_string(),
        content: "deny_credit".to_string(),
        signature: DecisionSignature { prince_veto: false, signature_bytes: vec![] },
        action_hash: [0; 32],
        is_critical: false,
        affects_rights: true,
        human_approval: None,
        decision_time: 100,
        explanation: Some("gradiente estocástico".to_string()),
    };

    assert_eq!(monitor.check_inv5_explainability(&bad_decision), false);
    assert_eq!(monitor.violation_log[0].invariant, "INV-5");
}
