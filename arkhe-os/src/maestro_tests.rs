use crate::maestro::spine::PsiState;
use crate::maestro::api_wrapper::PTPApiWrapper;
use crate::maestro::spine::MaestroSpine;
use crate::maestro::causality::BranchingEngine;
use crate::maestro::orchestrator::MaestroOrchestrator;
use crate::security::{XenoFirewall, XenoRiskLevel};
use std::sync::Arc;
use tokio::sync::RwLock;

#[tokio::test]
async fn test_xeno_firewall_safety() {
    let psi = PsiState::default();
    let safe_content = "This is a normal message about peer-to-peer cash.";
    assert_eq!(XenoFirewall::assess_risk(safe_content, &psi), XenoRiskLevel::Safe);

    let hazard_content = "danger ".repeat(600); // High density
    assert_eq!(XenoFirewall::assess_risk(&hazard_content, &psi), XenoRiskLevel::MemeticHazard);
}

#[tokio::test]
async fn test_xeno_firewall_critical() {
    let mut psi = PsiState::default();
    psi.current_coherence = 0.5; // Unstable
    let critical_content = "The stock market crash of 2029 will happen.";
    assert_eq!(XenoFirewall::assess_risk(critical_content, &psi), XenoRiskLevel::Critical);
}
