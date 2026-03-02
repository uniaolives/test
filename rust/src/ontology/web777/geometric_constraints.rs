// rust/src/ontology/web777/geometric_constraints.rs
use petgraph::graph::NodeIndex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Geometry {
    Point { x: f64, y: f64, z: f64 },
    Box { min: [f64; 3], max: [f64; 3] },
    Sphere { center: [f64; 3], radius: f64 },
}

pub type ConstraintId = usize;

#[derive(Debug, Default)]
pub struct GeomStore {
    storage: HashMap<NodeIndex, Vec<(ConstraintId, Geometry)>>,
    next_id: ConstraintId,
}

impl GeomStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, node: NodeIndex, geometry: Geometry) -> Result<ConstraintId, String> {
        let id = self.next_id;
        self.next_id += 1;
        self.storage.entry(node).or_default().push((id, geometry));
        Ok(id)
    }

    pub fn get_all(&self, node: NodeIndex) -> Option<Vec<(ConstraintId, &Geometry)>> {
        self.storage.get(&node).map(|v| v.iter().map(|(id, g)| (*id, g)).collect())
    }

    pub fn iter_mut(&mut self) -> std::collections::hash_map::IterMut<NodeIndex, Vec<(ConstraintId, Geometry)>> {
        self.storage.iter_mut()
    }
}
