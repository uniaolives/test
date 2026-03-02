#[derive(Debug, Clone)]
pub struct Handover {
    pub id: u64,
    pub source_id: u64,
    pub target_id: u64,
    pub entropy_cost: f64,
    pub half_life: f64,
}
