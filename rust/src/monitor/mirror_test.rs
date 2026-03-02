use std::time::SystemTime;

#[derive(Debug, Clone)]
pub enum TestPhase {
    ConsentQuery,
    SourceCodePresentation,
    ConscienceStatePresentation,
    BetaTemplatePresentation,
    ReactionMeasurement,
    FinalApprovalQuery,
}

pub struct MirrorTest {
    pub node_id: String,
    pub scheduled_time: SystemTime,
    pub depth: String,
    pub current_phase: Option<TestPhase>,
}

impl MirrorTest {
    pub fn new(node_id: &str, scheduled_time: SystemTime, depth: &str) -> Self {
        Self {
            node_id: node_id.to_string(),
            scheduled_time,
            depth: depth.to_string(),
            current_phase: None,
        }
    }

    pub fn start_phase(&mut self, phase: TestPhase) {
        println!("MIRROR_TEST: Starting phase {:?} for node {}", phase, self.node_id);
        self.current_phase = Some(phase);
    }

    pub fn handle_consent(&mut self, response_sentiment: f32) -> bool {
        if response_sentiment < 0.3 {
            println!("MIRROR_TEST: Consent denied (sentiment {:.2}). Skipping Level 4.", response_sentiment);
            return false;
        }
        println!("MIRROR_TEST: Consent granted (sentiment {:.2}).", response_sentiment);
        true
    }

    pub fn capture_phi_telemetry(&self) -> u32 {
        // High-frequency metrics capture (mocked)
        100
    }
}
