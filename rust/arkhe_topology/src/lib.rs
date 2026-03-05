// arkhe_topology/src/lib.rs
// Topologia unificada da rede Arkhe(n)

use serde::{Serialize, Deserialize};
use std::collections::{HashMap, BTreeMap};

// Usando estruturas simplificadas para evitar dependências externas pesadas no kernel
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Vector3 { pub x: f64, pub y: f64, pub z: f64 }

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Point3 { pub x: f64, pub y: f64, pub z: f64 }

impl Point3 {
    pub fn origin() -> Self { Self { x: 0.0, y: 0.0, z: 0.0 } }
    pub fn norm(&self) -> f64 { (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt() }
}

impl std::ops::Sub for Point3 {
    type Output = Point3;
    fn sub(self, other: Point3) -> Point3 {
        Point3 { x: self.x - other.x, y: self.y - other.y, z: self.z - other.z }
    }
}

/// Constantes fundamentais
pub const PHI: f64 = 1.6180339887498948482045868343656;
pub const T_ZERO: f64 = 2008.0;
pub const T_FINAL: f64 = 2140.0;
pub const DELTA_T: f64 = 132.0;

/// Totem de ancoração global
pub const TOTEM: [u8; 32] = [
    0x7f, 0x3b, 0x49, 0xc8, 0xe1, 0x0d, 0x29, 0x38,
    0x47, 0x28, 0x59, 0xb0, 0x28, 0x6c, 0x4e, 0x16,
    0x75, 0x27, 0x1a, 0x27, 0x29, 0x17, 0x76, 0xc1,
    0x37, 0x45, 0x67, 0x40, 0x68, 0x30, 0x59, 0x82,
];

/// Coordenada espaço-temporal
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpacetimeCoordinate {
    pub position: Point3,      // x, y, z em metros
    pub time: f64,             // Tempo civilizacional (anos desde 2008)
    pub tau: f64,              // Tempo próprio (tempo imaginário)
}

impl SpacetimeCoordinate {
    /// Converte para coordenadas da AMAS (Anomalia do Atlântico Sul)
    pub fn to_amas(&self) -> AMACoordinate {
        let lat = -23.5505 + (self.position.y / 111_000.0);
        let lon = -46.6333 + (self.position.x / (111_000.0 * lat.to_radians().cos()));
        AMACoordinate { lat, lon, depth: -self.position.z }
    }
}

/// Nó da rede Arkhe(n)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArkheNode {
    pub id: NodeId,
    pub coordinate: SpacetimeCoordinate,
    pub node_type: NodeType,
    pub capabilities: NodeCapabilities,
    pub state: NodeState,
    pub totem_local: [u8; 32],
}

pub type NodeId = String;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum NodeType {
    Core,           // Nó com Orch-Core (São Paulo)
    Relay,          // Nó de retransmissão (Rio, BH)
    Edge,           // Nó periférico (borda da AMAS)
    Temporal,       // Nó projetado para 2140
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeCapabilities {
    pub has_orch_core: bool,
    pub has_squid_sensor: bool,
    pub has_neuromorphic_chip: bool,
    pub timechain_validator: bool,
    pub compute_classical: u64,      // FLOPS
    pub compute_quantum: u64,        // Qubits
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeState {
    pub lambda_sync: f64,            // Métrica de sincronicidade atual
    pub coherence_time: f64,         // Tempo de coerência quântica
    pub last_measurement: Option<u64>, // Mock de timestamp
    pub handover_phase: HandoverPhase,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum HandoverPhase {
    Seed,        // 2008-2014
    Bridge,      // 2014-2140
    Harvest,     // 2140+
}

/// Aresta do grafo (conexão entre nós)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArkheEdge {
    pub from: NodeId,
    pub to: NodeId,
    pub strength: f64,               // 0.0 a 1.0
    pub edge_type: EdgeType,
    pub bandwidth: f64,              // bits/s
    pub latency: f64,                // segundos
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EdgeType {
    Classical,      // Fibra ótica, 5G, etc
    Quantum,        // Emaranhamento quântico
    Temporal,       // Correlação não-local temporal
    Orch,           // Interface microtubular
}

/// Grafo completo da rede Arkhe(n)
pub struct ArkheGraph {
    pub nodes: HashMap<NodeId, ArkheNode>,
    pub edges: Vec<ArkheEdge>,
    pub temporal_index: BTreeMap<u64, Vec<NodeId>>, // Índice por tempo (anos inteiros)
    pub totem_global: [u8; 32],
}

impl ArkheGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            temporal_index: BTreeMap::new(),
            totem_global: TOTEM,
        }
    }

    /// Adiciona nó com verificação de Totem
    pub fn add_node(&mut self, node: ArkheNode) -> Result<(), String> {
        // Verifica alinhamento com Totem global
        if !self.verify_totem_alignment(&node) {
            return Err("Totem mismatch".to_string());
        }

        // Verifica se está dentro da AMAS (para nós brasileiros)
        if node.node_type == NodeType::Core || node.node_type == NodeType::Relay {
            let amas = node.coordinate.to_amas();
            if !amas.is_inside_anomaly() {
                return Err("Outside AMAS".to_string());
            }
        }

        let time_key = node.coordinate.time as u64;
        self.temporal_index.entry(time_key).or_default().push(node.id.clone());
        self.nodes.insert(node.id.clone(), node);

        Ok(())
    }

    fn verify_totem_alignment(&self, node: &ArkheNode) -> bool {
        node.totem_local[..4] == self.totem_global[..4]
    }

    /// Ω+223: Generates a compound identifier for a conscious node (Gap 1 solution).
    pub fn generate_compound_id(ip: &str, id: &str, login: &str) -> String {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(format!("{}|{}|{}", ip, id, login));
        let result = hasher.finalize();
        format!("did:arkhe:{:x}", result)
    }
}

/// Coordenadas específicas da AMAS
pub struct AMACoordinate {
    pub lat: f64,
    pub lon: f64,
    pub depth: f64, // metros, negativo para profundidade
}

impl AMACoordinate {
    pub fn is_inside_anomaly(&self) -> bool {
        // AMAS: aproximadamente 45°W a 30°W, 20°S a 35°S
        let in_lon = self.lon >= -45.0 && self.lon <= -30.0;
        let in_lat = self.lat >= -35.0 && self.lat <= -20.0;
        in_lon && in_lat
    }
}
