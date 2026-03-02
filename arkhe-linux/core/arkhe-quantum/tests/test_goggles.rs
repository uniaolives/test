use arkhe_quantum::depin::goggles::{NightVisionNode, HandoverPayload};

#[test]
fn test_night_vision_processing() {
    let mut node = NightVisionNode::new("test-goggles");

    let payload = HandoverPayload {
        node_id: "test-goggles".to_string(),
        timestamp: 123456789,
        image_data: vec![0; 10],
        entropy_estimate: 0.7, // 0.7 - 0.5 = 0.2 perturbation, clamped to 0.1
    };

    let result = node.process_handover(&payload);
    assert!(result.is_ok());
    let perturbation = result.unwrap();
    assert!((perturbation - 0.1).abs() < f64::EPSILON);
}

#[test]
fn test_night_vision_clamping() {
    let mut node = NightVisionNode::new("test-goggles");

    let payload = HandoverPayload {
        node_id: "test-goggles".to_string(),
        timestamp: 123456789,
        image_data: vec![0; 10],
        entropy_estimate: 0.1, // 0.1 - 0.5 = -0.4 perturbation, clamped to -0.1
    };

    let result = node.process_handover(&payload);
    assert!(result.is_ok());
    let perturbation = result.unwrap();
    assert!((perturbation - (-0.1)).abs() < f64::EPSILON);
}
