use crate::physics::elasticity::ElasticSubdomain;

pub struct NodeCluster;
pub enum Topology {
    Hyperbolic { curvature: f64 },
}

pub struct InstaNode {
    pub physics: ElasticSubdomain,
}

impl NodeCluster {
    pub fn new(_nodes: usize, _topology: Topology) -> Self {
        Self
    }
    pub fn parallel_step<F>(&self, _f: F) where F: Fn(&InstaNode) {
    }
    pub fn elapsed_ms(&self) -> u64 {
        0
    }
    pub fn save_vtk(&self, _filename: String) {
    }
}

pub struct HSyncChannel;
impl NodeCluster {
    pub const sync_channel: HSyncChannel = HSyncChannel;
}
