// ledger.rs
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum GenesisEvent {
    STARNodeFormation {
        position: [f64; 3],
        timestamp: DateTime<Utc>,
    },
    PhotonCascade {
        start_node: [f64; 3],
        end_node: [f64; 3],
        wavelength_shift: f64,
        entropy_cost: f64,
    },
    MatterFormation {
        particle_type: String, // "electron", "positron", "graviton"
        position: [f64; 3],
        formation_energy: f64,
        phi_score: f64,
    },
    RBICycleCompletion {
        phase: String, // "converging" ou "diverging"
        duration_ms: u64,
        net_entropy: f64,
    },
}

pub struct GenesisLedger {
    events: Vec<GenesisEvent>,
}

impl GenesisLedger {
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    pub fn record(&mut self, event: GenesisEvent) {
        println!("📝 Registrando evento: {:?}", event);
        self.events.push(event);
    }

    /// Retorna todos os eventos de formação de matéria (elétrons, etc.)
    pub fn get_matter_formations(&self) -> Vec<&GenesisEvent> {
        self.events.iter()
            .filter(|e| matches!(e, GenesisEvent::MatterFormation { .. }))
            .collect()
    }

    /// Calcula a entropia total do sistema desde o início
    pub fn total_entropy(&self) -> f64 {
        self.events.iter()
            .map(|e| match e {
                GenesisEvent::PhotonCascade { entropy_cost, .. } => *entropy_cost,
                GenesisEvent::RBICycleCompletion { net_entropy, .. } => *net_entropy,
                _ => 0.0,
            })
            .sum()
    }
}
