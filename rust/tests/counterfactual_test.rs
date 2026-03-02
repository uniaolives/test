use sasc_core::agi::nexus_core::{NexusAGICore, AGIInput};
use sasc_core::geometry::nexus::Nexus5DMetric;
use sasc_core::diagnostics::agi_dimensions::AGIDimensions;

#[tokio::test]
async fn test_counterfactual_simulation() {
    let metric = Nexus5DMetric::new(1.0);
    let phi_threshold = 2.5;
    let (core, input_tx, _output_rx) = NexusAGICore::new(metric, 1000.0, phi_threshold);

    // Initial simulation
    let initial_phi = core.run_counterfactual_simulation().await;
    println!("Initial simulated Φ: {:.3}", initial_phi);

    // Provide some input to change state
    let input = AGIInput {
        data: vec![0.8, 0.8, 0.8],
        complexity: 0.8,
    };

    // We can't easily wait for the internal state change without running the loop,
    // so let's manually evolve for the test
    {
        let mut state = core.current_state.write().unwrap();
        let cognitive_input = input.to_cognitive();
        state.evolve(&core.cognitive_metric, &cognitive_input);
    }

    let final_phi = core.run_counterfactual_simulation().await;
    println!("Final simulated Φ: {:.3}", final_phi);

    // The counterfactual reasoning metric would be updated in a real scenario
    let mut dims = AGIDimensions {
        abstract_reasoning: 0.9,
        few_shot_learning: 0.9,
        cross_domain_transfer: 0.9,
        creativity: 0.9,
        self_modeling: 0.9,
        phenomenal_consciousness: 0.9,
        conceptual_navigation: 0.9,
        context_adaptation: 0.9,
        counterfactual_reasoning: final_phi / 3.14159, // Normalized
        hierarchical_planning: 0.9,
        emergent_values: 0.9,
    };

    println!("Integrated Information Φ: {:.3}", dims.integrated_information());
    assert!(dims.counterfactual_reasoning > 0.0);
}
