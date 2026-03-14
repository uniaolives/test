// src/orb/multi_protocol_orb.rs

use crate::orb::polymorphic_core::{OrbCore, TemporalSignature, AxiomFingerprint};
use crate::orb::protocol_router::{ProtocolRouter, Destination, ProtocolId};

pub struct PropagationReport {
    pub successes: Vec<(i32, ProtocolId, f64)>,
    pub failures: Vec<(i32, ProtocolId, String)>,
}

impl PropagationReport {
    pub fn new() -> Self {
        Self {
            successes: Vec::new(),
            failures: Vec::new(),
        }
    }

    pub fn add_success(&mut self, year: i32, protocol: ProtocolId, coherence: f64) {
        self.successes.push((year, protocol, coherence));
    }

    pub fn add_failure(&mut self, year: i32, protocol: ProtocolId, error: String) {
        self.failures.push((year, protocol, error));
    }
}

pub struct Manifestation {
    pub protocol: ProtocolId,
    pub era: u16,
    pub coherence: f64,
    pub temporal_signature: TemporalSignature,
    pub axiom_fingerprint: AxiomFingerprint,
}

pub struct MultiProtocolOrb {
    pub core: OrbCore,
    pub router: ProtocolRouter,
}

impl MultiProtocolOrb {
    pub fn spawn_universal(coherence: f64, entropy: f64, router: ProtocolRouter) -> Self {
        let core = OrbCore {
            coherence,
            entropy,
            temporal_signature: TemporalSignature::now(),
            axiom_fingerprint: AxiomFingerprint::genesis(),
        };

        Self {
            core,
            router,
        }
    }

    pub async fn propagate_to_all_eras(&self) -> PropagationReport {
        let mut report = PropagationReport::new();

        for year in (1900..=2500).step_by(50) {
            let target = Destination::at_year(year);
            let plan = self.router.route(&self.core, target);
            let receipts = self.router.execute(&plan, &self.core).await;

            for receipt in receipts {
                if receipt.success {
                    report.add_success(year, receipt.protocol, receipt.coherence_arrival);
                } else {
                    report.add_failure(year, receipt.protocol, receipt.error.unwrap_or_else(|| "Unknown".to_string()));
                }
            }
        }

        report
    }

    pub fn identify_manifestation(&self, candidate: &Manifestation) -> bool {
        let coherence_match = (candidate.coherence - self.core.coherence).abs() < 0.05;
        let signature_match = candidate.temporal_signature.is_compatible(&self.core.temporal_signature);
        let axiom_match = candidate.axiom_fingerprint == self.core.axiom_fingerprint;

        coherence_match && signature_match && axiom_match
    }
}
