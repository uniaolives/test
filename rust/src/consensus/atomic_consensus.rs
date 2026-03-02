use crate::substrate::{standing_wave_processor::{InterferenceProcessor, StandingWaveBit, DecoherenceError}, SubstrateGeometry, PhaseLock};
use crate::mesh_neuron::NodeId;
use num_complex::Complex64;

pub struct AtomicConsensus {
    pub processor: InterferenceProcessor,
    pub collective_mode: Option<StandingWaveBit>,
    pub network_geometry: SubstrateGeometry,
    pub coherence_history: Vec<f64>,
}

impl AtomicConsensus {
    pub fn new(geometry: SubstrateGeometry, _seed: &[u8; 32]) -> Self {
        let proc = InterferenceProcessor::initialize(
            geometry.clone(),
            (64, 64, 64),
            PhaseLock::schumann(),
        );

        Self {
            processor: proc,
            collective_mode: None,
            network_geometry: geometry,
            coherence_history: Vec::new(),
        }
    }

    pub fn propose_state(&mut self, state_hash: &[u8; 32]) -> Result<StandingWaveBit, DecoherenceError> {
        self.processor.encode_data(state_hash)?;
        let proposed = self.processor.standing_modes.last()
            .ok_or(DecoherenceError::ModeNotFound)?
            .clone();
        Ok(proposed)
    }

    pub fn receive_mode(&mut self, mode: &StandingWaveBit, _proposer: NodeId) -> Acceptance {
        if !self.network_geometry.allows_mode(mode) {
            return Acceptance::Reject(RejectReason::GeometricImpossibility);
        }

        if let Some(ref collective) = self.collective_mode {
            let overlap = self.processor.interfere_modes(
                self.processor.standing_modes.len().saturating_sub(1),
                0
            ).unwrap_or(Complex64::new(0.0, 0.0));

            let coherence = overlap.norm();
            self.coherence_history.push(coherence);

            if coherence > 0.9 {
                self.collective_mode = Some(self.combine_modes(collective, mode));
                Acceptance::Accept(AcceptanceWeight::Strong)
            } else if coherence > 0.5 {
                Acceptance::Accept(AcceptanceWeight::Weak)
            } else {
                Acceptance::Reject(RejectReason::Dissonance(coherence))
            }
        } else {
            self.collective_mode = Some(mode.clone());
            Acceptance::Accept(AcceptanceWeight::Genesis)
        }
    }

    fn combine_modes(&self, a: &StandingWaveBit, b: &StandingWaveBit) -> StandingWaveBit {
        StandingWaveBit {
            amplitude: (a.amplitude + b.amplitude) / 2.0,
            frequency: (a.frequency + b.frequency) / 2.0,
            wavenumber: a.wavenumber, // Simplified
            coherence_time: a.coherence_time.min(b.coherence_time),
        }
    }

    pub fn finalize_consensus(&self) -> ConsensusResult {
        if self.coherence_history.is_empty() {
            return ConsensusResult::Pending;
        }

        let avg_coherence = self.coherence_history.iter().sum::<f64>()
            / self.coherence_history.len() as f64;

        if avg_coherence > 0.8 {
            ConsensusResult::Achieved {
                coherence: avg_coherence,
            }
        } else {
            ConsensusResult::Failed
        }
    }
}

pub enum Acceptance {
    Accept(AcceptanceWeight),
    Reject(RejectReason),
}

pub enum AcceptanceWeight {
    Genesis,
    Strong,
    Weak,
}

pub enum RejectReason {
    GeometricImpossibility,
    Dissonance(f64),
}

pub enum ConsensusResult {
    Achieved { coherence: f64 },
    Pending,
    Failed,
}
