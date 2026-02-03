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
