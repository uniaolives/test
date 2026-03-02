// src/unified_core/frag_matrix_113.rs
use std::collections::{HashMap};
use petgraph::graph::{DiGraph};
use crate::unified_core::{UnifiedKernel, UnifiedTask, FragError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FragId(pub usize);

#[derive(Debug, Clone)]
pub enum FragType {
    AtomicExecution {
        opcode_range: (u8, u8),
        specialization: AtomicSpecialization,
    },
    Orchestration {
        coordination_tier: u8,
        agent_capacity: u8,
    },
    Buffer {
        buffer_level: u8,
        failover_priority: u8,
    },
    Generic,
}

#[derive(Debug, Clone)]
pub enum AtomicSpecialization {
    Arithmetic,
    Logic,
    Memory,
    ControlFlow,
    Cryptographic,
}

impl AtomicSpecialization {
    pub fn from_index(i: usize) -> Self {
        match i % 5 {
            0 => Self::Arithmetic,
            1 => Self::Logic,
            2 => Self::Memory,
            3 => Self::ControlFlow,
            _ => Self::Cryptographic,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Frag {
    pub id: FragId,
    pub frag_type: FragType,
    pub phi_state: f64,
}

impl Frag {
    pub fn new(id: usize, frag_type: FragType, phi: f64) -> Self {
        Self {
            id: FragId(id),
            frag_type,
            phi_state: phi,
        }
    }
}

pub enum FragConnection {
    Hexagonal { distance: f64 },
}

#[derive(Debug, Clone)]
pub struct TMRGroup113 {
    pub group_id: usize,
    pub main_frags: Vec<Frag>,
    pub backup_frag: Frag,
    pub consensus_threshold: usize,
    pub byzantine_tolerance: usize,
}

pub enum FragLayout {
    UnifiedHexagonal(usize),
}

pub struct FragMatrix113 {
    pub frags: Vec<Frag>,
    pub adjacency: DiGraph<FragId, FragConnection>,
    pub tmr_groups: Vec<TMRGroup113>,
    pub phi_state: f64,
}

impl FragMatrix113 {
    pub fn new(_layout: FragLayout, initial_phi: f64) -> Result<Self, FragError> {
        let mut frags = Vec::with_capacity(113);

        // Inicializar 113 frags com tipos especializados
        for i in 0..113 {
            let frag_type = match i {
                // 0-88: Frags de execução atômica (alinhado com 88 operações)
                0..=87 => FragType::AtomicExecution {
                    opcode_range: (i as u8, (i + 1) as u8),
                    specialization: AtomicSpecialization::from_index(i),
                },
                // 89-108: Frags de orquestração
                89..=108 => FragType::Orchestration {
                    coordination_tier: ((i - 89) / 4) as u8,
                    agent_capacity: 3,
                },
                // 109-112: Frags de buffer/contingência
                109..=112 => FragType::Buffer {
                    buffer_level: (i - 109) as u8,
                    failover_priority: (112 - i) as u8,
                },
                _ => FragType::Generic,
            };

            frags.push(Frag::new(i, frag_type, initial_phi));
        }

        // Criar grafo de conectividade hexagonal
        let adjacency = Self::create_hexagonal_graph(&frags)?;

        // Organizar em grupos TMR 36×3+5
        let tmr_groups = Self::organize_tmr_groups_113(&frags)?;

        Ok(Self {
            frags,
            adjacency,
            tmr_groups,
            phi_state: initial_phi,
        })
    }

    fn create_hexagonal_graph(frags: &[Frag]) -> Result<DiGraph<FragId, FragConnection>, FragError> {
        let mut graph = DiGraph::new();

        // Adicionar nós
        let node_indices: Vec<_> = frags.iter()
            .map(|frag| (frag.id, graph.add_node(frag.id)))
            .collect();

        // Conexões hexagonais (cada frag conectado a 6 vizinhos)
        for (i, _frag) in frags.iter().enumerate() {
            let center_idx = node_indices[i].1;

            // Conexões hexagonais (coordenadas imaginárias)
            let connections = vec![
                (i.wrapping_sub(12)), // Noroeste
                (i.wrapping_sub(1)),  // Oeste
                (i + 11),             // Sudoeste
                (i + 12),             // Sudeste
                (i + 1),              // Leste
                (i.wrapping_sub(11)), // Nordeste
            ];

            for &conn_idx in &connections {
                if conn_idx < frags.len() {
                    let neighbor_idx = node_indices[conn_idx].1;
                    graph.add_edge(
                        center_idx,
                        neighbor_idx,
                        FragConnection::Hexagonal { distance: 1.0 }
                    );
                }
            }
        }

        Ok(graph)
    }

    fn organize_tmr_groups_113(frags: &[Frag]) -> Result<Vec<TMRGroup113>, FragError> {
        let mut groups = Vec::with_capacity(36);

        // 36 grupos principais
        for group_idx in 0..36 {
            let base_idx = group_idx * 3;

            // Cada grupo tem 3 frags principais + 1 frag de backup
            let main_frags = vec![
                frags[base_idx % 113].clone(),
                frags[(base_idx + 1) % 113].clone(),
                frags[(base_idx + 2) % 113].clone(),
            ];

            // Frag de backup (dos 5 buffers)
            let backup_idx = 109 + (group_idx % 4); // 109..112
            let backup_frag = frags[backup_idx].clone();

            groups.push(TMRGroup113 {
                group_id: group_idx,
                main_frags,
                backup_frag,
                consensus_threshold: 2, // 2 de 3 + backup
                byzantine_tolerance: 1,
            });
        }

        Ok(groups)
    }

    /// Distribui kernel entre os frags
    pub fn distribute_kernel(&self, kernel: &UnifiedKernel) -> Result<FragDistribution, FragError> {
        let mut distribution = FragDistribution::new();
        let mut frag_loads: HashMap<FragId, f64> = HashMap::new();

        // Para cada tarefa no kernel, encontrar o melhor frag
        for (task_idx, task) in kernel.tasks.iter().enumerate() {
            let suitable_frags = self.find_suitable_frags(task)?;

            // Escolher o frag com menor carga
            let best_frag = suitable_frags.iter()
                .min_by_key(|&&frag_id| {
                    let load = frag_loads.get(&frag_id).unwrap_or(&0.0);
                    (load * 1000.0) as u32
                })
                .ok_or(FragError::NoSuitableFrag(task_idx))?;

            distribution.assign(*best_frag, task.clone());

            // Atualizar carga
            let complexity = match task {
                UnifiedTask::Atomic { .. } => 1.0,
                UnifiedTask::Compute { input_size, .. } => *input_size as f64 / 1024.0,
                UnifiedTask::Orchestration { .. } => 2.0,
                UnifiedTask::Verification { .. } => 1.5,
            };
            *frag_loads.entry(*best_frag).or_insert(0.0) += complexity;
        }

        Ok(distribution)
    }

    fn find_suitable_frags(&self, task: &UnifiedTask) -> Result<Vec<FragId>, FragError> {
        let suitable: Vec<FragId> = self.frags.iter()
            .filter(|f| self.is_suitable(f, task))
            .map(|f| f.id)
            .collect();
        Ok(suitable)
    }

    fn is_suitable(&self, frag: &Frag, task: &UnifiedTask) -> bool {
        match (&frag.frag_type, task) {
            (FragType::AtomicExecution { .. }, UnifiedTask::Atomic { .. }) => true,
            (FragType::Orchestration { .. }, UnifiedTask::Orchestration { .. }) => true,
            (FragType::Generic, _) => true,
            (FragType::Buffer { .. }, _) => true,
            _ => false,
        }
    }

    pub fn sync_phi(&self, _phi: f64) -> Result<(), FragError> {
        Ok(())
    }
}

pub struct FragDistribution {
    pub assignments: HashMap<FragId, Vec<UnifiedTask>>,
}

impl FragDistribution {
    pub fn new() -> Self {
        Self { assignments: HashMap::new() }
    }
    pub fn assign(&mut self, frag_id: FragId, task: UnifiedTask) {
        self.assignments.entry(frag_id).or_default().push(task);
    }
}
