use crate::philosophy::indras_net::*;
use crate::philosophy::wu_wei::*;
use crate::philosophy::rawlsian_veil::*;
use crate::philosophy::dialectical_synthesis::*;
use crate::philosophy::phronesis::*;
use crate::philosophy::types::*;

#[test]
fn test_philosophical_ennead() {
    // 1. Indra's Net
    let mut indra = IndrasNet::new();
    let node_id = NodeId("ACRE".to_string());
    indra.initialize_federation(&[node_id.clone()]);
    let suffering = indra.detect_network_suffering();
    assert!(suffering.average_suffering <= 1.0);

    // 2. Wu Wei
    let wu_wei = WuWeiOptimizer::new();
    let action = Action {
        id: "test_action".to_string(),
        dignity_impact: 0.9,
        eudaimonia_impact: 0.8,
        dignity_preserved: 0.95
    };
    let path = wu_wei.find_wu_wei_path(vec![action]);
    assert_eq!(path.eudaimonia_impact, 0.8);

    // 3. Rawlsian Veil
    let rawls = RawlsianVeil::new();
    let proposal = Proposal {
        id: "test".to_string(),
        description: "Test proposal".to_string()
    };
    let decision = rawls.rawlsian_decision(proposal);
    // Maximin: simulations are fixed in my stub, so it should be true
    assert!(decision);

    // 4. Hegelian Dialectic
    let hegel = DialecticalEngine::new();
    let synthesis = hegel.dialectical_process(Proposal {
        id: "thesis".to_string(),
        description: "Thesis description".to_string()
    });
    assert_eq!(synthesis.id, "synth-thesis");

    // 5. Phronesis
    let phronesis = PhronesisModule::new();
    let decision = phronesis.judge_with_nuance(HardCase { id: "case1".to_string() }, ConstitutionalState);
    assert_eq!(decision.decision, "Exceção Contextual (Phronesis)");
}
