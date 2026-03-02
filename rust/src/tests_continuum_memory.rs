use crate::memory::continuum_system::*;
use crate::entropy::VajraEntropyMonitor;
use crate::governance::SASCCathedral;
use std::sync::Arc;

#[tokio::test]
async fn test_continuum_memory_system() {
    let vajra = Arc::new(VajraEntropyMonitor::global().clone_monitor());
    let cathedral = Arc::new(SASCCathedral);

    let mut agent = SecureAgent::new(
        "TestAgent".to_string(),
        10,
        50,
        Some(vajra),
        Some(cathedral),
    );

    let prompt = "How to treat stage-4 cyber-cancer?";
    let context = AgentContext;

    // First query - Cache Miss
    let res1 = agent.process(prompt, &context).await.unwrap();
    assert!(!res1.cached);
    assert_eq!(res1.cache_source, "MISS");

    // Second query - Cache Hit
    let res2 = agent.process(prompt, &context).await.unwrap();
    assert!(res2.cached);
    assert_eq!(res2.cache_source, "MTM");

    // Verify stats
    let stats = agent.cms.get_stats();
    assert_eq!(stats.total_queries, 2);
    assert_eq!(stats.mtm_hits, 1);
    assert_eq!(stats.misses, 1);
}

#[tokio::test]
async fn test_semantic_similarity() {
    let mut cms = SemanticCMS::new(
        "SemanticTest".to_string(),
        10,
        50,
        0.87,
        None,
        None,
    );

    let prompt1 = "Quantum attestation protocol";
    let response1 = "Protocol active";
    let metadata = CacheMetadata {
        agent_name: "Test".to_string(),
        phi_score: 0.99,
        lyapunov_delta: 0.0,
        attack_family: None,
        ghost_density: 0.0,
        quantum_proof_valid: true,
    };

    cms.store(prompt1, response1, metadata, false).unwrap();

    // Mock similarity is 1.0 because embeddings are all zeros in this mock
    let prompt2 = "Quantum attestation system";
    let (res, similarity, source) = cms.query(prompt2, false).unwrap().expect("Semantic hit expected");

    assert_eq!(source, "SEMANTIC");
    assert!(similarity >= 0.87);
}
