// arkhe-quantum/src/manifold.rs

use std::collections::HashMap;
use crate::crypto::NodeKeys;
use crate::qkd::QuantumTunnel;

pub struct Node {
    pub id: String,
    pub entropy_val: f64,
}

impl Node {
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
        nodes.insert("self".to_string(), Node { id: "self".to_string(), entropy_val: 0.5 });

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
}
