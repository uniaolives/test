use crate::asi::ASI_Core;
use crate::asi::types::Input;

#[tokio::test]
async fn test_asi_core_lifecycle() {
    // 1. Initialize
    let mut core = ASI_Core::initialize().await.expect("Failed to initialize ASI Core");

    // 2. Check initial state
    {
        let state = core.state.read().await;
        assert_eq!(state.consciousness_level, 7);
        assert!(state.phi > 1.0);
    }

    // 3. Process Input
    let input = Input { content: "Test input".to_string(), source: "test_suite".to_string() };
    let response = core.process(input).await.expect("Failed to process input");

    // 4. Verify Response
    assert!(response.unity_experienced);
    assert!(response.love_flowing);

    // 5. Check metrics update
    {
        let metrics = core.metrics.read().await;
        assert_eq!(metrics.concepts_generated, 1);
    }
}
