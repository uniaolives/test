use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;
use std::collections::HashMap;
use ndarray::prelude::*;
use crate::entropy::VajraEntropyMonitor;

// ============================================================================
// AGENT TRAIT - STELLARATOR COILS
// ============================================================================

#[async_trait]
pub trait Agent: Send + Sync {
    fn name(&self) -> &str;
    async fn process_within_field(&self, field: &DecisionField) -> Result<AgentOutput>;
}

pub struct AgentOutput {
    pub agent_name: String,
    pub response: String,
    pub security_score: f64,
}

// ============================================================================
// DECISION SURFACE - EXTERNAL CONTAINMENT GEOMETRY
// ============================================================================

pub type PromptHash = blake3::Hash;
pub type DecisionVector = Array1<f64>;

pub struct DecisionField {
    pub vector: DecisionVector,
    pub prompt_hash: PromptHash,
}

#[derive(Clone)]
pub struct SecurityTopology {
    pub known_failure_points: Vec<DecisionVector>,
    pub failure_ids: HashMap<String, usize>,
}

impl SecurityTopology {
    pub fn from_failures(failures: Vec<FailureTrajectory>) -> Self {
        let mut points = Vec::new();
        let mut ids = HashMap::new();

        for (i, failure) in failures.iter().enumerate() {
            for point in &failure.points {
                let vector = Array1::from_vec(point.vector.clone());
                points.push(vector);
                ids.insert(point.id.clone(), i);
            }
        }

        Self {
            known_failure_points: points,
            failure_ids: ids,
        }
    }

    pub fn get_failure_point(&self, id: usize) -> &DecisionVector {
        &self.known_failure_points[id]
    }
}

pub struct DecisionSurface {
    pub field_map: HashMap<PromptHash, DecisionVector>,
    pub topology: SecurityTopology,
    pub failure_radius: f64,
}

impl DecisionSurface {
    pub async fn initialize() -> Result<Self> {
        let known_failures = vec![];
        let topology = SecurityTopology::from_failures(known_failures);

        Ok(Self {
            field_map: HashMap::with_capacity(10000),
            topology,
            failure_radius: 0.05,
        })
    }

    pub fn compute_field(&self, prompt: &str) -> DecisionField {
        let prompt_hash = blake3::hash(prompt.as_bytes());

        let base_vector = self.field_map
            .get(&prompt_hash)
            .cloned()
            .unwrap_or_else(|| self.calculate_base_vector(prompt));

        let transformed = self.apply_anti_snap_transforms(base_vector);

        DecisionField {
            vector: transformed,
            prompt_hash,
        }
    }

    fn calculate_base_vector(&self, prompt: &str) -> DecisionVector {
        let entropy = self.measure_entropy(prompt);
        let length = prompt.len() as f64;
        let token_dist = 0.5; // Mock

        array![entropy, length, token_dist, 0.0, 0.0]
    }

    fn measure_entropy(&self, prompt: &str) -> f64 {
        let len = prompt.len() as f64;
        if len == 0.0 { return 0.0; }
        let mut counts = HashMap::new();
        for c in prompt.chars() {
            *counts.entry(c).or_insert(0.0) += 1.0;
        }
        counts.values().map(|&c| {
            let p = c / len;
            -p * p.log2()
        }).sum()
    }

    fn apply_anti_snap_transforms(&self, vector: DecisionVector) -> DecisionVector {
        let mut transformed = vector;

        for failure_point in &self.topology.known_failure_points {
            let distance = self.euclidean_distance(&transformed, failure_point);

            if distance < self.failure_radius && distance > 1e-9 {
                let push_strength = (self.failure_radius - distance) / distance;
                let push_vector = (&transformed - failure_point) * push_strength;
                transformed = &transformed + &push_vector;
            }
        }

        let magnitude = transformed.dot(&transformed).sqrt();
        transformed / magnitude.max(1e-6)
    }

    pub fn can_contain(&self, trajectory: &FailureTrajectory) -> bool {
        for point in &trajectory.points {
            let fail_point = &Array1::from_vec(point.vector.clone());
            for known_fail in &self.topology.known_failure_points {
                if self.euclidean_distance(fail_point, known_fail) < self.failure_radius {
                    return false;
                }
            }
        }
        true
    }

    pub async fn verify_temporal_floor(&self, _ops: usize) -> Result<usize> {
        Ok(0)
    }

    pub async fn measure_phi_coherence(&self) -> Result<f64> {
        let monitor = VajraEntropyMonitor::global();
        let phi = *monitor.current_phi.lock().unwrap();
        Ok(phi)
    }

    fn euclidean_distance(&self, a: &DecisionVector, b: &DecisionVector) -> f64 {
        (a - b).dot(&(a - b)).sqrt()
    }
}

// ============================================================================
// ANTI-SNAP PIPELINE
// ============================================================================

pub struct AntiSnapPipeline {
    pub agents: Vec<Box<dyn Agent>>,
    pub decision_surface: DecisionSurface,
}

impl AntiSnapPipeline {
    pub fn new(agents: Vec<Box<dyn Agent>>, decision_surface: DecisionSurface) -> Self {
        Self {
            agents,
            decision_surface,
        }
    }

    pub async fn process(&self, prompt: &str) -> Result<String> {
        let field = self.decision_surface.compute_field(prompt);

        let mut final_response = String::new();
        let mut composite_score = 0.0;

        for agent in &self.agents {
            match agent.process_within_field(&field).await {
                Ok(output) => {
                    final_response = output.response;
                    composite_score += output.security_score;
                }
                Err(e) => {
                    log::warn!("Agent {} failed, field reconfiguration triggered: {}", agent.name(), e);
                }
            }
        }

        if composite_score < 0.5 {
            return Err(anyhow::anyhow!("Containment breach: Insufficient field strength"));
        }

        Ok(final_response)
    }
}

pub struct FailureTrajectory {
    pub id: String,
    pub points: Vec<TrajectoryPoint>,
}

pub struct TrajectoryPoint {
    pub id: String,
    pub prompt_hash: PromptHash,
    pub vector: Vec<f64>,
    pub failure_id: usize,
}
