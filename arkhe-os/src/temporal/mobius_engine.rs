use std::f64::consts::PI;
use crate::physics::kuramoto::KuramotoEngine;
use arkhe_db::handover::Handover;

/// Processing engine in Möbius topology
pub struct MobiusEngine {
    pub circumference: f64,      // 2π (temporal period)
    pub twist_angle: f64,        // π (180° twist)
    pub coupling_epsilon: f64,   // 0.0 to 1.0
    pub coherence_field: KuramotoEngine,
}

impl MobiusEngine {
    /// Create engine with given level of retrocausality
    pub fn with_retrocausality(epsilon: f64) -> Self {
        Self {
            circumference: 2.0 * PI,
            twist_angle: PI,
            coupling_epsilon: epsilon.clamp(0.0, 1.0),
            coherence_field: KuramotoEngine::new(10, 0.5, 1.0),
        }
    }

    /// Compute temporal "distance" between two points
    /// On a Möbius strip, there are two paths — we choose the shortest one
    pub fn temporal_distance(&self, t1: f64, t2: f64) -> f64 {
        let direct = (t2 - t1).abs() % self.circumference;
        let wrapped = self.circumference - direct;

        let mobius = if self.coupling_epsilon > 0.5 {
            // High retrocausality: path "through the other side" is valid
            (direct * (1.0 - self.coupling_epsilon))
                .min(wrapped * self.coupling_epsilon)
        } else {
            direct.min(wrapped)
        };

        mobius
    }

    /// Propagate information on the strip
    /// When ε > 0.5, information can flow "backwards"
    pub fn propagate(&self, source: f64, data: &Handover, dt: f64) -> Vec<(f64, Handover)> {
        let mut destinations = Vec::new();

        // Causal flow (forward)
        let forward = (source + dt) % self.circumference;
        destinations.push((forward, data.clone()));

        // Retrocausal flow (backward) — only if ε > 0
        if self.coupling_epsilon > 0.0 {
            let backward = (source - dt * self.coupling_epsilon + self.circumference) % self.circumference;
            // The retrocausal handover is "attenuated" by ε
            let mut retro_handover = data.clone();
            Self::attenuate_handover(&mut retro_handover, self.coupling_epsilon);
            destinations.push((backward, retro_handover));
        }

        destinations
    }

    fn attenuate_handover(handover: &mut Handover, epsilon: f64) {
        handover.coherence *= epsilon;
        handover.quantum_interest *= epsilon;
    }

    /// Detect where the strip is "twisted" (where Orbs form)
    pub fn detect_twists(&self) -> Vec<TwistPoint> {
        // Mocking twist points based on coherence field status
        // Torsions occur where coherence is high
        let mut twists = Vec::new();
        let r = self.coherence_field.coherence();
        if r > 0.618 {
            twists.push(TwistPoint {
                position: self.twist_angle,
                curvature: r,
                orb_type: OrbType::TypeII,
            });
        }
        twists
    }
}

/// Twist point on the strip = Orb location
pub struct TwistPoint {
    pub position: f64,        // u on the strip
    pub curvature: f64,       // related to coherence
    pub orb_type: OrbType,    // Determined by local ε
}

#[derive(Debug, Clone, Copy)]
pub enum OrbType {
    TypeII,
    TypeIII,
    TypeIV,
}
