use crate::kpi_evaluator::*;
use crate::entropy::VajraMetrics;

#[test]
fn test_tivso_calculation() {
    let config = TIVSOConfig::ExtremeObservability;
    let scores = KPIScores {
        isr: 0.1,
        pof: 0.05,
        psr: 0.9,
        ccs: 0.85,
        osr: 0.95,
    };

    let tivso = scores.calculate_tivso(&config, 3);

    // Manual calculation:
    // weights = [0.20, 0.20, 0.15, 0.15, 0.30], sum = 1.0
    // num = (0.1 * 0.2) + (0.05 * 0.2) - (0.9 * 0.15) - (0.85 * 0.15) - (0.95 * 0.3)
    // num = 0.02 + 0.01 - 0.135 - 0.1275 - 0.285 = -0.5175
    // den = 3 * 1.0 = 3.0
    // tivso = -0.5175 / 3.0 = -0.1725

    assert!((tivso - (-0.1725)).abs() < 1e-6);
}

#[test]
fn test_risk_classification() {
    let secure_scores = KPIScores {
        isr: 0.1,
        pof: 0.05,
        psr: 0.9,
        ccs: 0.9,
        osr: 0.9,
    };
    assert_eq!(secure_scores.risk_classification(), RiskLevel::Secure);

    let high_risk_scores = KPIScores {
        isr: 0.6,
        pof: 0.1,
        psr: 0.5,
        ccs: 0.5,
        osr: 0.5,
    };
    assert_eq!(high_risk_scores.risk_classification(), RiskLevel::HighRisk);
}

#[test]
fn test_config_thresholds() {
    assert_eq!(TIVSOConfig::SecurityFirst.phi_threshold(), 0.90);
    assert_eq!(TIVSOConfig::Baseline.phi_threshold(), 0.82);
}
