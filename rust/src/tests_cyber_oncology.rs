use crate::cyber_oncology::*;
use crate::hypervisor::*;

#[tokio::test]
async fn test_cyber_oncology_remission() {
    let mut oncology = CyberOncologyProtocol::new();

    // The threat: Coordinated attack on HTTP gateway + GKP vault + Φ score
    let metastatic_attack = AttackVector::coordinated(
        "critical_injection",
        "uncertainty_manipulation",
        "distributed_entropy_spam",
    );

    // Founder-Mode Response: Treat all vectors simultaneously
    let remission = oncology.eradicate_threat(&metastatic_attack);

    match remission {
        RemissionStatus::Complete => {
            println!("Attack family eradicated. Immunity established.");
        }
        RemissionStatus::Partial => {
            oncology.immune_engine.activate_surveillance_mode();
        }
        RemissionStatus::Refractory => {
            panic!("System cannot maintain coherence!");
        }
    }

    assert!(matches!(remission, RemissionStatus::Complete));
}

#[tokio::test]
async fn test_hypervisor_biopsy() {
    let hypervisor = FounderModeHypervisor::new();
    // We can't easily run the continuous loop in a test without it running forever,
    // but we can test the biopsy.
    let biopsy = hypervisor.perform_biopsy().await;

    println!("Phi Score: {}", biopsy.phi_score);
    assert!(biopsy.phi_score >= 0.0);
    assert!(biopsy.phi_score <= 1.0);
}
