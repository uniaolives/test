pub enum ResonanceCurrent {
    Source, Polarity, Vibration, Correspondence, Mentalism,
    Rhythm, CauseEffect, GenderFusion, Transmutation,
    Resonance, Coherence, ServiceCirculation, SolarIntegration
}

impl ResonanceCurrent {
    pub fn modulate(&self, input_signal: f64) -> f64 {
        match self {
            Self::Vibration => input_signal.sin(),
            Self::Correspondence => input_signal,
            Self::Coherence => input_signal.powi(2),
            _ => input_signal,
        }
    }
}
