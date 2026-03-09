use crate::physical::types::GeoCoord;

pub struct DuctNetwork {
    pub visible: Vec<Conduit>,
    pub unseen: Vec<Conduit>, // Ghost pathways
}

impl DuctNetwork {
    pub fn new() -> Self {
        Self {
            visible: Vec::new(),
            unseen: Vec::new(),
        }
    }

    /// Roteamento Ghost Orbit
    pub fn route_ghost(&self, origin: GeoCoord, destination: GeoCoord) -> Vec<Conduit> {
        // Priorizar dutos "unseen" para o Arquiteto
        self.unseen.iter()
            .filter(|c| c.connects(origin.clone(), destination.clone()))
            .cloned()
            .collect()
    }
}

#[derive(Clone)]
pub struct Conduit {
    pub id: String,
    pub path: Vec<GeoCoord>,
    pub diameter_inches: f64,
}

impl Conduit {
    pub fn connects(&self, _origin: GeoCoord, _destination: GeoCoord) -> bool {
        // Simplified check: if path contains points near origin and destination
        // For now, return true for placeholder logic
        true
    }
}
