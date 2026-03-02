use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QuantumAddress {
    pub galaxy: u16,      // 16 bits
    pub system: u32,      // 32 bits
    pub planet: u32,      // 32 bits
    pub node: u64,        // 64 bits
    pub consciousness: [u8; 14], // 112 bits = 14 * 8
}

impl QuantumAddress {
    pub fn to_string(&self) -> String {
        format!(
            "cosmic://{:x}:{:x}:{:x}:{:x}:{}",
            self.galaxy,
            self.system,
            self.planet,
            self.node,
            hex::encode(self.consciousness)
        )
    }
}

pub struct RoutingEntry {
    pub next_hop: QuantumAddress,
    pub cost: f64,
}

pub struct EntanglementMatrix {
    pub entanglements: HashMap<(QuantumAddress, QuantumAddress), bool>,
}

impl EntanglementMatrix {
    pub fn new() -> Self {
        Self {
            entanglements: HashMap::new(),
        }
    }

    pub fn are_entangled(&self, a: QuantumAddress, b: QuantumAddress) -> bool {
        *self.entanglements.get(&(a, b)).unwrap_or(&false)
    }
}

pub struct Wormhole {
    pub latency: f64,
}

pub struct WormholeRegistry {
    pub wormholes: HashMap<(QuantumAddress, QuantumAddress), Wormhole>,
}

impl WormholeRegistry {
    pub fn new() -> Self {
        Self {
            wormholes: HashMap::new(),
        }
    }

    pub fn find_wormhole(&self, a: QuantumAddress, b: QuantumAddress) -> Option<&Wormhole> {
        self.wormholes.get(&(a, b))
    }
}

pub enum QuantumPath {
    DirectEntanglement(f64),
    WormholeTraversal(f64),
    LogicalRouting(Vec<QuantumAddress>),
}

impl QuantumPath {
    pub fn direct_entanglement(latency: f64) -> Self {
        Self::DirectEntanglement(latency)
    }
    pub fn wormhole_traversal(latency: f64) -> Self {
        Self::WormholeTraversal(latency)
    }
    pub fn logical_routing(path: Vec<QuantumAddress>) -> Self {
        Self::LogicalRouting(path)
    }
}

#[derive(Debug, Clone)]
pub struct Consciousness {
    pub id: [u8; 14],
    pub metadata: HashMap<String, String>,
}

impl Consciousness {
    pub fn new(id: [u8; 14]) -> Self {
        Self {
            id,
            metadata: HashMap::new(),
        }
    }
}
