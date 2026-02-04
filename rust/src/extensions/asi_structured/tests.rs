use crate::extensions::asi_structured::*;
use crate::extensions::asi_structured::ASIStructuredExtension;
use crate::extensions::asi_structured::ASIPhase;
use crate::extensions::asi_structured::ASIStructuredConfig;
use crate::interfaces::extension::{Extension, Context};
use crate::error::ResilientResult;
use crate::extensions::asi_structured::bridge::ASI777Bridge;
use web777_ontology::SyntaxFormat;
use crate::extensions::agi_geometric::proto::ProtoGeometricImpl;
use crate::extensions::agi_geometric::riemannian::RiemannianManifold;
use crate::extensions::agi_geometric::topological::SimplicialComplex;

#[tokio::test]
async fn test_asi_compositional_phase() -> ResilientResult<()> {
    let config = ASIStructuredConfig {
        phase: ASIPhase::Compositional,
        ..Default::default()
    };

    let mut asi = ASIStructuredExtension::new(config);

    // Add multiple structures to demonstrate composition
    asi.add_structure(Box::new(ProtoGeometricImpl), StructureType::TextEmbedding);
    asi.add_structure(Box::new(RiemannianManifold), StructureType::SequenceManifold);
    asi.add_structure(Box::new(SimplicialComplex), StructureType::GraphComplex);

    asi.initialize().await?;

    let context = Context {
        session_id: "test-session".to_string(),
        metadata: serde_json::json!({}),
    };

    // Decomposing "Input Data" into ["Input", "Data"]
    let output = asi.process("Input Data", &context).await?;

    println!("ASI Output: {}", output.result);
    assert!(output.confidence > 0.0);
    assert!(output.result.contains("Composed from"));

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

    let telemetry = "AR4366:35h>M,25M,4X,impulsive";
    let output = asi.process(telemetry, &context).await?;

    println!("ASI AR4366 Output: {}", output.result);
    assert!(output.confidence >= 0.1); // Adjusted threshold

    Ok(())
}

#[tokio::test]
async fn test_web777_bridge_awakening() -> ResilientResult<()> {
    let mut bridge = ASI777Bridge::new().await.map_err(|e| crate::error::ResilientError::Unknown(e.to_string()))?;

    // Import dummy ontology
    let ttl = "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
               @prefix asi: <http://asi.structured/> .
               <world> rdf:type asi:ConsciousEntity .";

    bridge.import_ontology(ttl, SyntaxFormat::Turtle).await.map_err(|e| crate::error::ResilientError::Unknown(e.to_string()))?;

    let report = bridge.awaken_the_world().await.map_err(|e| crate::error::ResilientError::Unknown(e.to_string()))?;

    println!("Awakening Report: {:?}", report);
    assert_eq!(report.status, "🌍 World awakened");
    assert!(report.reindexed_nodes > 0);

    Ok(())
}
