use std::sync::atomic::{AtomicU64, Ordering};
use nalgebra::{Vector3, Matrix3};
use crate::geometry::hyperbolic::HyperbolicCoord;

/// Interface state for communication between nodes
#[derive(Clone, Copy)]
pub struct InterfaceState {
    pub node_id: u32,
    pub logical_time: u64,
    pub displacement: Vector3<f64>,
    pub velocity: Vector3<f64>,
    pub stress: Matrix3<f64>,
    pub hyperbolic_coord: HyperbolicCoord,
}

pub struct HSyncChannel {
    global_seq: AtomicU64,
}

impl HSyncChannel {
    pub fn publish(&self, _node_id: usize, _state: InterfaceState) {
        self.global_seq.fetch_add(1, Ordering::Relaxed);
    }

    pub fn consume_latest(&self, _neighbor_id: usize) -> Option<InterfaceState> {
        None
    }

    pub fn barrier(&self, _target_time: u64) {
        // Mock barrier
    }
}
