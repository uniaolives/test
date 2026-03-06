use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct TimelineBranch {
    pub id: Uuid,
    pub parent_id: Option<Uuid>,
    pub narrative: String,
    pub phi_q: f64,
    pub berry_phase: f64,
}

pub struct BranchingEngine {
    pub active_branches: Vec<TimelineBranch>,
    pub max_divergence: f64,
}

impl BranchingEngine {
    pub fn new(max_divergence: f64) -> Self {
        Self {
            active_branches: Vec::new(),
            max_divergence,
        }
    }

    pub fn prune_branches(&mut self) {
        // Se a fase de Berry desviar de π/2 (90°) além da tolerância,
        // o ramo é considerado um paradoxo e removido.
        self.active_branches.retain(|b| {
            let phase_error = (b.berry_phase - std::f64::consts::PI / 2.0).abs();
            phase_error < self.max_divergence
        });
    }

    pub fn fork(&mut self, parent_id: Option<Uuid>, new_narrative: String, phi: f64) -> Uuid {
        let new_id = Uuid::new_v4();
        let new_branch = TimelineBranch {
            id: new_id,
            parent_id,
            narrative: new_narrative,
            phi_q: phi,
            berry_phase: std::f64::consts::PI / 2.0, // Default to coherent phase for now
        };
        self.active_branches.push(new_branch);
        new_id
    }
}
