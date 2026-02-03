use std::collections::HashMap;
use petgraph::graph::NodeIndex;
use serde::{Deserialize, Serialize};
use nalgebra::Vector3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConstraintId(pub u64);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Geometry {
    Point(Vector3<f64>),
    Box { min: Vector3<f64>, max: Vector3<f64> },
    Circle { center: Vector3<f64>, radius: f64 },
}

#[derive(Debug, Default)]
pub struct GeomStore {
    constraints: HashMap<NodeIndex, Vec<(ConstraintId, Geometry)>>,
    next_id: u64,
}

impl GeomStore {
    pub fn new() -> Self {
        Self {
            constraints: HashMap::new(),
            next_id: 0,
        }
    }

    pub fn insert(&mut self, node: NodeIndex, geom: Geometry) -> Result<ConstraintId, String> {
        let id = ConstraintId(self.next_id);
        self.next_id += 1;
        self.constraints.entry(node).or_default().push((id, geom));
        Ok(id)
    }

    pub fn get_all(&self, node: NodeIndex) -> Option<Vec<(ConstraintId, &Geometry)>> {
        self.constraints.get(&node).map(|v| {
            v.iter().map(|(id, g)| (*id, g)).collect()
        })
    }
}
