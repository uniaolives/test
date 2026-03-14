// src/orb/protocol_router.rs

use std::any::Any;
use std::time::Duration;
use crate::orb::polymorphic_core::{OrbCore, ProtocolEncoder, Reach};

pub type ProtocolId = usize;

pub struct Destination {
    pub timestamp: u64,
    pub distance_m: f64,
}

impl Destination {
    pub fn at_year(year: i32) -> Self {
        let years_from_1970 = year - 1970;
        Self {
            timestamp: (years_from_1970 as u64).wrapping_mul(365 * 24 * 3600),
            distance_m: 0.0,
        }
    }

    pub fn distance(&self) -> f64 {
        self.distance_m
    }
}

pub struct Hop {
    pub protocol: ProtocolId,
}

pub struct RoutePlan {
    pub hops: Vec<Hop>,
    pub redundancy: usize,
}

pub struct TransmissionReceipt {
    pub protocol: ProtocolId,
    pub success: bool,
    pub coherence_arrival: f64,
    pub error: Option<String>,
}

pub struct ProtocolRouter {
    pub encoders: Vec<Box<dyn ProtocolEncoder>>,
}

impl ProtocolRouter {
    pub fn calculate_fitness(&self, orb: &OrbCore, encoder: &dyn ProtocolEncoder, target: &Destination) -> f64 {
        let bandwidth = encoder.bandwidth();
        let coherence_preserved = if bandwidth.is_infinite() {
            1.0
        } else {
            bandwidth / (bandwidth + orb.entropy + 1e-9)
        };

        let latency_penalty = 1.0 / (1.0 + encoder.latency().as_secs_f64());

        let reach_match = match (encoder.reach(), target.distance()) {
            (Reach::NonLocal, _) => 1.0,
            (Reach::Global, _) => 0.9,
            (Reach::Regional(r), d) if r > d => 0.7,
            (Reach::Local(r), d) if r > d => 0.5,
            (Reach::LineOfSight, _) => 0.3,
            _ => 0.0,
        };

        coherence_preserved * latency_penalty * reach_match * orb.coherence
    }

    pub fn route(&self, orb: &OrbCore, target: Destination) -> RoutePlan {
        let mut candidates: Vec<(ProtocolId, f64)> = self.encoders.iter()
            .enumerate()
            .map(|(id, encoder)| {
                let fitness = self.calculate_fitness(orb, encoder.as_ref(), &target);
                (id, fitness)
            })
            .filter(|(_, fitness)| *fitness > 0.0)
            .collect();

        // Sort by fitness descending
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let hops = candidates.into_iter()
            .map(|(protocol, _)| Hop { protocol })
            .collect();

        RoutePlan {
            hops,
            redundancy: 3,
        }
    }

    pub async fn execute(&self, plan: &RoutePlan, orb: &OrbCore) -> Vec<TransmissionReceipt> {
        let mut receipts = Vec::new();

        for hop in &plan.hops[..plan.redundancy.min(plan.hops.len())] {
            let encoder = &self.encoders[hop.protocol];
            let result = encoder.encode_to_any(orb);

            receipts.push(TransmissionReceipt {
                protocol: hop.protocol,
                success: result.is_ok(),
                coherence_arrival: orb.coherence * 0.99, // Mock loss
                error: result.err().map(|e| format!("{:?}", e)),
            });
        }

        receipts
    }
}
