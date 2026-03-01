// src/www/frag_matrix_116.rs
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use petgraph::graph::{UnGraph, NodeIndex};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use crate::www::protocol_dispatch::{WebProtocol, WebRequest, WebResponse};
use crate::www::global_orbit::Region;

#[derive(Debug, thiserror::Error)]
pub enum MatrixError {
    #[error("Assignment mismatch: {0} assigned, expected {1}")]
    AssignmentMismatch(usize, usize),
    #[error("Unsupported protocol: {0:?}")]
    UnsupportedProtocol(WebProtocol),
    #[error("No frag found for strategy")]
    NoFragFound,
    #[error("Other matrix error: {0}")]
    Other(String),
}

pub type FragId = usize;

pub enum FragConnection {
    WebMesh {
        latency_ms: f64,
        bandwidth_gbps: f64,
    },
}

/// Matriz de 116 frags para a World Wide Web
pub struct FragMatrix116 {
    frags: Vec<WebFrag>,
    topology: UnGraph<FragId, FragConnection>,
    protocol_assignments: HashMap<WebProtocol, Vec<FragId>>,
    region_assignments: HashMap<Region, Vec<FragId>>,
    load_balancer: Arc<LoadBalancer>,
}

pub struct WebFrag {
    pub id: FragId,
    pub frag_type: WebFragType,
    pub phi: f64,
}

impl WebFrag {
    pub fn new(id: FragId, frag_type: WebFragType, phi: f64) -> Self {
        Self { id, frag_type, phi }
    }

    pub async fn process_web_request(&self, request: &WebRequest) -> Result<WebResponse, MatrixError> {
        // Dummy implementation
        Ok(WebResponse::new())
    }
}

pub enum WebFragType {
    HttpServer {
        version: String,
        tls_enabled: bool,
        max_connections: usize,
    },
    WebSocketServer {
        secure: bool,
        compression: bool,
        message_size_limit: usize,
    },
    QuicServer {
        version: String,
        multiplexing: bool,
        zero_rtt: bool,
    },
    GlobalRouter {
        region: Region,
        bgp_peers: usize,
        routing_table_size: usize,
    },
}

pub struct LoadBalancer {
    phi: f64,
}

impl LoadBalancer {
    pub fn new(phi: f64) -> Result<Self, MatrixError> {
        Ok(Self { phi })
    }

    pub fn select_frag(&self, candidates: &[FragId], _frags: &[WebFrag]) -> Result<FragId, MatrixError> {
        if candidates.is_empty() {
            return Err(MatrixError::NoFragFound);
        }
        // Round robin or random for dummy
        Ok(candidates[0])
    }

    pub fn update_load(&self, _frag_id: FragId, _load: f64) -> Result<(), MatrixError> {
        Ok(())
    }
}

impl FragMatrix116 {
    pub fn new(initial_phi: f64) -> Result<Self, MatrixError> {
        let mut frags = Vec::with_capacity(116);

        // Inicializar 116 frags com especializações web
        for i in 0..116 {
            let frag_type = match i {
                // 0-39: Frags HTTP/HTTPS (40 frags)
                0..=39 => WebFragType::HttpServer {
                    version: if i < 20 { "1.1".to_string() } else { "2".to_string() },
                    tls_enabled: i % 2 == 0,
                    max_connections: 10000,
                },

                // 40-79: Frags WebSocket (40 frags)
                40..=79 => WebFragType::WebSocketServer {
                    secure: i % 3 == 0,
                    compression: true,
                    message_size_limit: 16 * 1024 * 1024,
                },

                // 80-99: Frags QUIC/HTTP3 (20 frags)
                80..=99 => WebFragType::QuicServer {
                    version: "draft-29".to_string(),
                    multiplexing: true,
                    zero_rtt: i % 2 == 0,
                },

                // 100-115: Frags de roteamento global (16 frags)
                100..=115 => WebFragType::GlobalRouter {
                    region: Self::assign_region(i),
                    bgp_peers: 100,
                    routing_table_size: 100000,
                },
                _ => unreachable!(),
            };

            frags.push(WebFrag::new(i, frag_type, initial_phi));
        }

        // Criar topologia em malha com redundância
        let topology = Self::create_mesh_topology(&frags)?;

        // Atribuir protocolos aos frags
        let protocol_assignments = Self::assign_protocols(&frags)?;

        // Atribuir regiões
        let region_assignments = Self::assign_regions(&frags)?;

        Ok(Self {
            frags,
            topology,
            protocol_assignments,
            region_assignments,
            load_balancer: Arc::new(LoadBalancer::new(initial_phi)?),
        })
    }

    fn assign_region(i: usize) -> Region {
        match i % 6 {
            0 => Region::SouthAmerica,
            1 => Region::NorthAmerica,
            2 => Region::Europe,
            3 => Region::Asia,
            4 => Region::Africa,
            _ => Region::Oceania,
        }
    }

    fn create_mesh_topology(frags: &[WebFrag]) -> Result<UnGraph<FragId, FragConnection>, MatrixError> {
        let mut graph = UnGraph::new_undirected();

        // Adicionar nós
        let node_indices: Vec<_> = frags.iter()
            .map(|frag| (frag.id, graph.add_node(frag.id)))
            .collect();

        // Conexões em malha com grau médio 8
        for i in 0..frags.len() {
            let current_idx = node_indices[i].1;

            // Conexões para balanceamento de carga
            let connections = vec![
                (i + 1) % frags.len(),
                (i + 2) % frags.len(),
                (i + 29) % frags.len(), // Número primo para distribuição
                (i + 53) % frags.len(),
                (i + 87) % frags.len(),
                (i + frags.len() - 1) % frags.len(),
                (i + frags.len() - 29) % frags.len(),
                (i + frags.len() - 53) % frags.len(),
            ];

            for &conn_idx in &connections {
                if let Some(&(_, neighbor_idx)) = node_indices.get(conn_idx) {
                    graph.add_edge(current_idx, neighbor_idx, FragConnection::WebMesh {
                        latency_ms: 1.0,
                        bandwidth_gbps: 10.0,
                    });
                }
            }
        }

        Ok(graph)
    }

    fn assign_protocols(frags: &[WebFrag]) -> Result<HashMap<WebProtocol, Vec<FragId>>, MatrixError> {
        let mut assignments = HashMap::new();

        // Protocolos principais
        let protocols = vec![
            (WebProtocol::Http, 0..20),
            (WebProtocol::Https, 20..40),
            (WebProtocol::WebSocket, 40..60),
            (WebProtocol::WebSocketSecure, 60..80),
            (WebProtocol::Http3, 80..100),
            (WebProtocol::AtProto, 100..104),
            // ... preencher até 104 protocolos
        ];

        for (protocol, range) in protocols {
            for i in range {
                assignments.entry(protocol.clone())
                    .or_insert_with(Vec::new)
                    .push(frags[i].id);
            }
        }

        // Fill remaining for 104 protocols
        for i in 0..104 {
            let protocol = WebProtocol::from_index(i);
            if !assignments.contains_key(&protocol) {
                // Assign some frag
                assignments.entry(protocol)
                    .or_insert_with(Vec::new)
                    .push(frags[i % 116].id);
            }
        }

        Ok(assignments)
    }

    fn assign_regions(frags: &[WebFrag]) -> Result<HashMap<Region, Vec<FragId>>, MatrixError> {
        let mut assignments = HashMap::new();
        for frag in frags {
            let region = Self::assign_region(frag.id);
            assignments.entry(region)
                .or_insert_with(Vec::new)
                .push(frag.id);
        }
        Ok(assignments)
    }

    /// Roteia requisição através da matriz web
    pub async fn route_web_request(
        &self,
        request: &WebRequest,
        strategy: RoutingStrategy,
    ) -> Result<WebResponse, MatrixError> {
        // 1. Determinar protocolo
        let protocol = request.protocol();

        // 2. Selecionar frags para o protocolo
        let candidate_frags = self.protocol_assignments.get(&protocol)
            .ok_or(MatrixError::UnsupportedProtocol(protocol.clone()))?;

        // 3. Aplicar estratégia de roteamento
        let selected_frag = match strategy {
            RoutingStrategy::LoadBalanced => {
                self.load_balancer.select_frag(candidate_frags, &self.frags)?
            }
            RoutingStrategy::Geographic(region) => {
                self.select_frag_by_region(candidate_frags, &region)?
            }
            RoutingStrategy::LatencyOptimized => {
                self.select_frag_by_latency(candidate_frags, "TODO")?
            }
        };

        // 4. Processar no frag selecionado
        let frag = &self.frags[selected_frag];
        let response = frag.process_web_request(request).await?;

        // 5. Atualizar métricas
        self.load_balancer.update_load(selected_frag, 1.0)?;

        Ok(response)
    }

    fn select_frag_by_region(&self, candidates: &[FragId], region: &Region) -> Result<FragId, MatrixError> {
        for &id in candidates {
            if Self::assign_region(id) == *region {
                return Ok(id);
            }
        }
        self.load_balancer.select_frag(candidates, &self.frags)
    }

    fn select_frag_by_latency(&self, candidates: &[FragId], _source: &str) -> Result<FragId, MatrixError> {
        self.load_balancer.select_frag(candidates, &self.frags)
    }
}

pub enum RoutingStrategy {
    LoadBalanced,
    Geographic(Region),
    LatencyOptimized,
}
