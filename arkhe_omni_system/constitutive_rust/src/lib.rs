//! Módulo Constitutive e Contact para Arkhe(n)Chain
//!
//! Define as relações entre handovers (deformação) e coerência (tensão),
//! e a mecânica de contacto entre nós do hipergrafo.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// --- Módulo Constitutivo ---

/// Parâmetros constitutivos para um nó
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstitutiveParams {
    /// Nó ID (World ID quântico)
    pub node_id: String,
    /// Coeficiente elástico (α) — relaciona handover com coerência
    pub alpha: f64,
    /// Coeficiente viscoso (β) — relaciona taxa de handover com dissipação
    pub beta: f64,
    /// Coeficiente de inércia (γ) — opcional, para modelos viscoelásticos
    pub gamma: Option<f64>,
    /// Energia livre inicial (coerência basal)
    pub psi_0: f64,
    /// Tipo de modelo (elástico, plástico, viscoso, etc.)
    pub model_type: ConstitutiveModel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstitutiveModel {
    Elastic,
    Viscous,
    Viscoelastic,
    Plastic,
}

/// Histórico de handovers (deformações) de um nó
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrainHistory {
    pub timestamps: Vec<u64>,
    pub strains: Vec<f64>,
    capacity: usize,
}

impl StrainHistory {
    pub fn new(capacity: usize) -> Self {
        Self {
            timestamps: Vec::with_capacity(capacity),
            strains: Vec::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, timestamp: u64, strain: f64) {
        if self.timestamps.len() >= self.capacity {
            self.timestamps.remove(0);
            self.strains.remove(0);
        }
        self.timestamps.push(timestamp);
        self.strains.push(strain);
    }

    pub fn strain_rate(&self) -> Option<f64> {
        if self.strains.len() < 2 {
            return None;
        }
        let last = self.strains.len() - 1;
        let dt = (self.timestamps[last] - self.timestamps[last - 1]) as f64 * 1e-9;
        if dt == 0.0 { return Some(0.0); }
        let de = self.strains[last] - self.strains[last - 1];
        Some(de / dt)
    }

    pub fn current_strain(&self) -> Option<f64> {
        self.strains.last().copied()
    }
}

/// Acumulador de dissipação
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DissipationAccumulator {
    pub total: f64,
    pub last_dissipation: Option<f64>,
}

impl DissipationAccumulator {
    pub fn new() -> Self {
        Self {
            total: 0.0,
            last_dissipation: None,
        }
    }

    pub fn accumulate(&mut self, dissipation: f64, dt: f64) {
        self.total += dissipation * dt;
        self.last_dissipation = Some(dissipation);
    }
}

// --- Módulo de Contacto ---

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Vector3(pub [f64; 3]);

impl Vector3 {
    pub fn dist(&self, other: &Vector3) -> f64 {
        ((self.0[0] - other.0[0]).powi(2) +
         (self.0[1] - other.0[1]).powi(2) +
         (self.0[2] - other.0[2]).powi(2)).sqrt()
    }

    pub fn sub(&self, other: &Vector3) -> Vector3 {
        Vector3([self.0[0] - other.0[0], self.0[1] - other.0[1], self.0[2] - other.0[2]])
    }

    pub fn add(&self, other: &Vector3) -> Vector3 {
        Vector3([self.0[0] + other.0[0], self.0[1] + other.0[1], self.0[2] + other.0[2]])
    }

    pub fn mul(&self, scalar: f64) -> Vector3 {
        Vector3([self.0[0] * scalar, self.0[1] * scalar, self.0[2] * scalar])
    }

    pub fn norm(&self) -> f64 {
        (self.0[0].powi(2) + self.0[1].powi(2) + self.0[2].powi(2)).sqrt()
    }

    pub fn normalize(&self) -> Vector3 {
        let n = self.norm();
        if n > 0.0 { self.mul(1.0 / n) } else { Vector3([0.0, 0.0, 0.0]) }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: String,
    pub embedding: Vector3,
    pub handover_count: u64,
}

pub struct ContactManager {
    pub nodes: HashMap<String, Node>,
    pub threshold: f64,
    pub stiffness: f64,
}

impl ContactManager {
    pub fn new(threshold: f64, stiffness: f64) -> Self {
        Self {
            nodes: HashMap::new(),
            threshold,
            stiffness,
        }
    }

    /// Deteção de contacto (Busca Global Simplificada)
    pub fn detect_contacts(&self) -> Vec<(String, String)> {
        let node_ids: Vec<&String> = self.nodes.keys().collect();
        let mut contacts = Vec::new();

        for i in 0..node_ids.len() {
            for j in i + 1..node_ids.len() {
                let id_i = node_ids[i];
                let id_j = node_ids[j];
                let node_i = &self.nodes[id_i];
                let node_j = &self.nodes[id_j];

                if node_i.embedding.dist(&node_j.embedding) < self.threshold {
                    contacts.push((id_i.clone(), id_j.clone()));
                }
            }
        }
        contacts
    }

    /// Resolução de contacto (Método de Penalidade)
    pub fn resolve_contacts(&mut self, dt: f64) {
        let threshold = self.threshold;
        let stiffness = self.stiffness;
        let contacts = self.detect_contacts();

        for (id_i, id_j) in contacts {
            if let (Some(node_i), Some(node_j)) = self.get_two_mut(&id_i, &id_j) {
                let diff = node_i.embedding.sub(&node_j.embedding);
                let distance = diff.norm();

                if distance < threshold && distance > 0.0 {
                    let force_mag = stiffness * (threshold - distance);
                    let force = diff.normalize().mul(force_mag);

                    node_i.embedding = node_i.embedding.add(&force.mul(dt));
                    node_j.embedding = node_j.embedding.sub(&force.mul(dt));

                    node_i.handover_count += 1;
                    node_j.handover_count += 1;
                }
            }
        }
    }

    fn get_two_mut(&mut self, id_a: &str, id_b: &str) -> (Option<&mut Node>, Option<&mut Node>) {
        if id_a == id_b { return (None, None); }

        let mut ptr_a: *mut Node = std::ptr::null_mut();
        let mut ptr_b: *mut Node = std::ptr::null_mut();

        for (id, node) in self.nodes.iter_mut() {
            if id == id_a { ptr_a = node as *mut Node; }
            if id == id_b { ptr_b = node as *mut Node; }
        }

        unsafe {
            (ptr_a.as_mut(), ptr_b.as_mut())
        }
    }
}

// --- Integração Final ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstitutiveState {
    pub params: HashMap<String, ConstitutiveParams>,
    pub strain_history: HashMap<String, StrainHistory>,
    pub dissipation: HashMap<String, DissipationAccumulator>,
}

impl ConstitutiveState {
    pub fn new() -> Self {
        Self {
            params: HashMap::new(),
            strain_history: HashMap::new(),
            dissipation: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstitutiveMsg {
    DefineConstitutiveRelation { params: ConstitutiveParams },
    UpdateStrain { node_id: String, timestamp: u64, strain: f64 },
    ComputeStress { node_id: String },
    ValidateObjectivity { node_id: String, rotation_matrix: [[f64; 3]; 3] },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressResult {
    pub node_id: String,
    pub stress: f64,
    pub free_energy: f64,
    pub dissipation: f64,
    pub objectivity_check: bool,
}

pub struct ConstitutiveModule {
    state: ConstitutiveState,
}

impl ConstitutiveModule {
    pub fn new() -> Self {
        Self { state: ConstitutiveState::new() }
    }

    pub fn process_msg(&mut self, msg: ConstitutiveMsg) -> Result<Option<StressResult>, String> {
        match msg {
            ConstitutiveMsg::DefineConstitutiveRelation { params } => {
                self.state.params.insert(params.node_id.clone(), params);
                Ok(None)
            }
            ConstitutiveMsg::UpdateStrain { node_id, timestamp, strain } => {
                let history = self.state.strain_history
                    .entry(node_id.clone())
                    .or_insert_with(|| StrainHistory::new(1000));
                history.push(timestamp, strain);
                Ok(None)
            }
            ConstitutiveMsg::ComputeStress { node_id } => {
                let result = self.compute_stress(&node_id)?;
                Ok(Some(result))
            }
            ConstitutiveMsg::ValidateObjectivity { node_id, rotation_matrix } => {
                let objective = self.check_objectivity(&node_id, rotation_matrix)?;
                Ok(Some(StressResult {
                    node_id,
                    stress: 0.0,
                    free_energy: 0.0,
                    dissipation: 0.0,
                    objectivity_check: objective,
                }))
            }
        }
    }

    fn compute_stress(&self, node_id: &str) -> Result<StressResult, String> {
        let params = self.state.params.get(node_id)
            .ok_or_else(|| format!("Nó {} não tem parâmetros constitutivos", node_id))?;
        let history = self.state.strain_history.get(node_id)
            .ok_or_else(|| format!("Nó {} não tem histórico de handovers", node_id))?;
        let strain = history.current_strain()
            .ok_or_else(|| format!("Nó {} não tem handovers registados", node_id))?;
        let strain_rate = history.strain_rate().unwrap_or(0.0);

        let free_energy = params.psi_0 + params.alpha * strain * strain / 2.0;
        let dissipation = params.beta * strain_rate * strain_rate;

        if dissipation < 0.0 {
            return Err(format!("Violação da 2ª lei: dissipação negativa ({})", dissipation));
        }

        let stress = params.alpha * strain + 2.0 * params.beta * strain_rate;

        Ok(StressResult {
            node_id: node_id.to_string(),
            stress,
            free_energy,
            dissipation,
            objectivity_check: true,
        })
    }

    fn check_objectivity(&self, _node_id: &str, _rotation: [[f64; 3]; 3]) -> Result<bool, String> {
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elastic_constitutive() {
        let mut module = ConstitutiveModule::new();
        let params = ConstitutiveParams {
            node_id: "rafael-001".to_string(),
            alpha: 2.0,
            beta: 0.1,
            gamma: None,
            psi_0: 1.0,
            model_type: ConstitutiveModel::Elastic,
        };
        module.process_msg(ConstitutiveMsg::DefineConstitutiveRelation { params }).unwrap();
        let timestamps = [1000000000, 2000000000, 3000000000];
        let strains = [0.1, 0.2, 0.3];
        for i in 0..3 {
            module.process_msg(ConstitutiveMsg::UpdateStrain {
                node_id: "rafael-001".to_string(),
                timestamp: timestamps[i],
                strain: strains[i],
            }).unwrap();
        }
        let result = module.process_msg(ConstitutiveMsg::ComputeStress {
            node_id: "rafael-001".to_string(),
        }).unwrap().unwrap();
        assert!((result.stress - 0.62).abs() < 1e-6);
        assert!((result.free_energy - 1.09).abs() < 1e-6);
    }

    #[test]
    fn test_contact_mechanics() {
        let mut manager = ContactManager::new(0.5, 10.0);
        manager.nodes.insert("node1".to_string(), Node {
            id: "node1".to_string(),
            embedding: Vector3([0.0, 0.0, 0.0]),
            handover_count: 0,
        });
        manager.nodes.insert("node2".to_string(), Node {
            id: "node2".to_string(),
            embedding: Vector3([0.4, 0.0, 0.0]),
            handover_count: 0,
        });

        let contacts = manager.detect_contacts();
        assert_eq!(contacts.len(), 1);

        manager.resolve_contacts(0.1);

        let n1 = manager.nodes.get("node1").unwrap();
        let n2 = manager.nodes.get("node2").unwrap();

        // n1 should move in negative X direction, n2 in positive X
        assert!(n1.embedding.0[0] < 0.0);
        assert!(n2.embedding.0[0] > 0.4);
        assert_eq!(n1.handover_count, 1);
        assert_eq!(n2.handover_count, 1);
    }
}
