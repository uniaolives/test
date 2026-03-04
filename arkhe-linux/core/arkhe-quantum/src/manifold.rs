// arkhe-quantum/src/manifold.rs

use std::collections::HashMap;
use std::time::Instant;
use crate::crypto::NodeKeys;
use crate::qkd::QuantumTunnel;
use crate::QuantumState;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum NodeState {
    Unlicensed,
    Licensing,
    Active,
    Inhibited,
}

pub struct Node {
    pub id: String,
    pub entropy_val: f64,
    pub state: NodeState,
    pub last_activation: Option<Instant>,
    pub handover_count: u64,
}

impl Node {
    pub fn new(id: String) -> Self {
        Self {
            id,
            entropy_val: 0.5,
            state: NodeState::Unlicensed,
            last_activation: None,
            handover_count: 0,
        }
    }

    pub fn entropy(&self) -> f64 {
        self.entropy_val
    }
}

pub struct GlobalManifold {
    pub nodes: HashMap<String, Node>,
    pub node_keys: NodeKeys,
    pub tunnels: HashMap<String, QuantumTunnel>,
}

impl GlobalManifold {
    pub fn new() -> Self {
        let mut nodes = HashMap::new();
        let mut self_node = Node::new("self".to_string());
        self_node.state = NodeState::Active;
        nodes.insert("self".to_string(), self_node);

        Self {
            nodes,
            node_keys: NodeKeys::generate(),
            tunnels: HashMap::new(),
        }
    }

    pub fn get_self_node(&self) -> Option<&Node> {
        self.nodes.get("self")
    }

    pub fn get_self_node_mut(&mut self) -> Option<&mut Node> {
        self.nodes.get_mut("self")
    }

    pub fn license_node(&mut self, node_id: &str) -> bool {
        if let Some(node) = self.nodes.get_mut(node_id) {
            if node.state == NodeState::Active {
                return false;
            }
            node.state = NodeState::Licensing;
            tracing::info!("Node {} is now in Licensing state (Pre-RC)", node_id);
            true
        } else {
            false
        }
    }

    pub fn activate_node(&mut self, node_id: &str) -> bool {
        if let Some(node) = self.nodes.get_mut(node_id) {
            if node.state == NodeState::Licensing {
                node.state = NodeState::Active;
                node.last_activation = Some(Instant::now());
                tracing::info!("Node {} is now Active (CDK/DDK trigger)", node_id);
                true
            } else {
                false
            }
        } else {
            false
        }
    }
}
