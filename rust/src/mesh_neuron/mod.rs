use crate::maat::flagellar_dynamics::{PropulsionMode, NodeId};
use crate::maat::scenarios::network_congestion::{AttackVector, RoutingMode};

pub struct MeshNeuron {
    pub id: NodeId,
}

impl MeshNeuron {
    pub fn compromise(&mut self, _vector: AttackVector) {}
    pub fn disable_screw_propulsion(&mut self) {}
    pub fn enable_screw_propulsion(&mut self, _enabled: bool) {}
    pub fn set_routing_mode(&mut self, _mode: RoutingMode) {}
    pub fn set_density_threshold(&mut self, _threshold: f64) {}
    pub fn activate_ubuntu_collective(&mut self) {}
}

pub struct UbuntuWeightedConsensus;
