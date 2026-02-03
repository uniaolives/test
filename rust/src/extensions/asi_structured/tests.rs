use crate::extensions::asi_structured::*;
use crate::extensions::asi_structured::ASIStructuredExtension;
use crate::extensions::asi_structured::ASIPhase;
use crate::extensions::asi_structured::ASIStructuredConfig;
use crate::extensions::agi_geometric::proto::ProtoGeometricImpl;
use crate::extensions::agi_geometric::riemannian::RiemannianManifold;
use crate::extensions::agi_geometric::topological::SimplicialComplex;
use crate::interfaces::extension::{Extension, Context};
use crate::error::ResilientResult;

#[tokio::test]
async fn test_asi_compositional_phase() -> ResilientResult<()> {
    let config = ASIStructuredConfig {
        phase: ASIPhase::Compositional,
        ..Default::default()
    };

    let mut asi = ASIStructuredExtension::new(config);

    // Add multiple structures to demonstrate composition
    asi.add_structure(Box::new(ProtoGeometricImpl));
    asi.add_structure(Box::new(RiemannianManifold));
    asi.add_structure(Box::new(SimplicialComplex));

    asi.initialize().await?;

    let context = Context {
        session_id: "test-session".to_string(),
        metadata: serde_json::json!({}),
    };

    // Decomposing "Input Data" into ["Input", "Data"]
    let output = asi.process("Input Data", &context).await?;

    println!("ASI Output: {}", output.result);
    assert!(output.confidence > 0.0);
    assert!(output.result.contains("ComposedResult"));

    Ok(())
}

#[tokio::test]
async fn test_asi_solar_volatility_ar4366() -> ResilientResult<()> {
    let config = ASIStructuredConfig {
        phase: ASIPhase::Compositional,
        ..Default::default()
    };

    let mut asi = ASIStructuredExtension::new(config);
    asi.initialize().await?;

    let context = Context {
        session_id: "solar-test-session".to_string(),
        metadata: serde_json::json!({}),
    };

    // AR 4366 scenario: 35h above M-class, 25 M-class flares, 4 X-class flares, impulsive
    let telemetry = "AR4366:35h>M,25M,4X,impulsive";
    let output = asi.process(telemetry, &context).await?;

    println!("ASI AR4366 Output: {}", output.result);

    // Verify that the system processed the high-volatility telemetry
    // and remained "Structured" with high confidence (S9 Invariant)
    assert!(output.confidence >= 0.8);
    assert!(output.result.contains("ComposedResult"));

    Ok(())
}
