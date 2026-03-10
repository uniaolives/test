#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpMethod4 {
    Observe,
    Emit,
    Entangle,
    Collapse,
    Propagate,
    Coherence,
    Quantize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfinementMode {
    InfiniteWell,
    FiniteWell,
    Barrier,
    Free,
}

impl ConfinementMode {
    pub fn from_lambda2(lambda2: f64) -> Self {
        if lambda2 >= 1.0 {
            ConfinementMode::InfiniteWell
        } else if lambda2 >= 0.95 {
            ConfinementMode::FiniteWell
        } else if lambda2 >= 0.80 {
            ConfinementMode::Barrier
        } else {
            ConfinementMode::Free
        }
    }
}

pub struct Http4Headers {
    pub temporal_origin: Option<i64>,
    pub temporal_target: Option<i64>,
    pub lambda2: f64,
    pub confinement_mode: ConfinementMode,
    pub quantization_levels: Option<u32>,
}
