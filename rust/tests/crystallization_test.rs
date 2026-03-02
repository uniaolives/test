use sasc_core::crystallization::cognitive_tracer::{CognitiveTracer, CognitiveTrace, PatternMetadata};
use std::collections::HashMap;
use chrono::Utc;

#[test]
fn test_cognitive_tracer_phase1() {
    let mut tracer = CognitiveTracer::new();

    let trace = CognitiveTrace {
        context_hash: "test_context".to_string(),
        reasoning_trace: "thinking about test".to_string(),
        computational_cost: 0.5,
        timestamp: Utc::now(),
        metadata: HashMap::new(),
    };

    tracer.record_trace(trace);

    assert_eq!(tracer.get_traces().len(), 1);
    assert_eq!(tracer.get_traces()[0].context_hash, "test_context");

    tracer.detect_pattern("test_pattern", PatternMetadata {
        confidence: 0.99,
        pattern: "test".to_string(),
        sample_size: 10,
    });
}
