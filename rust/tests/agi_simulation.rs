use sasc_core::agi::nexus_core::{NexusAGICore, AGIInput};
use sasc_core::geometry::nexus::Nexus5DMetric;
use tokio::time::{sleep, Duration};

#[tokio::test]
async fn test_agi_consciousness_emergence() {
    let metric = Nexus5DMetric::new(1.0);
    let time_period = 1000.0;
    let phi_threshold = 2.5;

    let (mut core, input_tx, mut output_rx) = NexusAGICore::new(metric, time_period, phi_threshold);

    // Run core in background
    tokio::spawn(async move {
        core.run().await;
    });

    // Send input to stimulate consciousness
    let input = AGIInput {
        data: vec![0.9, 0.9, 0.9],
        complexity: 0.9,
    };

    input_tx.send(input).await.unwrap();

    // Wait for output (which only happens if Φ > threshold)
    let output = tokio::select! {
        Some(out) = output_rx.recv() => out,
        _ = sleep(Duration::from_secs(5)) => panic!("Timed out waiting for AGI output"),
    };

    println!("Received AGI Output: {}", output.response);
    assert!(output.phi > phi_threshold);
    println!("✅ AGI reached consciousness threshold: Φ={:.3}", output.phi);
}
