#[derive(Debug, Clone)]
pub enum SelfModification {
    AddLayer(String),
    PruneConnections(f64),
    ChangeActivation(String),
    RewireTopology(String),
    NoOp,
}
