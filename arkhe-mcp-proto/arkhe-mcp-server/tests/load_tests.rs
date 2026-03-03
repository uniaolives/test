use std::sync::Arc;
use tokio::sync::Semaphore;
use std::time::{Duration, Instant};
use serde_json::json;

mod common;

#[tokio::test]
async fn test_concurrent_handover_flood() {
    let mcp_server = Arc::new(common::TestMcpServer::start_test_mcp_server().await);
    let semaphore = Arc::new(Semaphore::new(100));

    let start = Instant::now();
    let mut handles = vec![];

    for _ in 0..100 { // Reduced to 100 for faster local test
        let permit = semaphore.clone().acquire_owned().await.unwrap();
        let server = mcp_server.clone();

        handles.push(tokio::spawn(async move {
            let _permit = permit;
            server.call_tool("get_phi_state", json!({})).await;
        }));
    }

    for h in handles {
        h.await.unwrap();
    }

    let elapsed = start.elapsed();
    assert!(elapsed < Duration::from_secs(30));
}
