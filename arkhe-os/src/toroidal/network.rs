// arkhe-os/src/toroidal/network.rs
//! Rede Toroidal para GNSS Shield v4.0
//! Propaga Orbs respeitando a topologia Yin-Yang para detecção de spoofing.

use std::collections::HashMap;
use std::f64::consts::PI;
use crate::toroidal::yin_yang::{PNTInfo, ToroidalMode};

pub struct ToroidalNetwork {
    /// Fases conhecidas dos nós vizinhos (ID -> Fase [0, 2π])
    pub neighbor_phases: HashMap<String, f64>,
    /// Limiar de coerência para aceitação (φ = 0.618)
    pub coherence_threshold: f64,
}

impl ToroidalNetwork {
    pub fn new() -> Self {
        Self {
            neighbor_phases: HashMap::new(),
            coherence_threshold: 0.618033988749895, // Golden Ratio
        }
    }

    /// Calcula a distância geodésica no círculo toroidal
    pub fn toroidal_distance(&self, p1: f64, p2: f64) -> f64 {
        let diff = (p1 - p2).abs();
        diff.min(2.0 * PI - diff)
    }

    /// Valida um nó baseado na divergência de fase
    /// Se a distância no espaço de fase for muito grande, é um spoofer em potencial.
    pub fn validate_node(&self, node_id: &str, observed_phase: f64) -> bool {
        if let Some(&expected_phase) = self.neighbor_phases.get(node_id) {
            let dist = self.toroidal_distance(observed_phase, expected_phase);
            // Distância máxima permitida baseada no limiar de coerência
            let max_dist = PI * (1.0 - self.coherence_threshold);
            dist < max_dist
        } else {
            // Novo nó: aceita mas monitora
            true
        }
    }

    /// Propaga um Orb de PNT pela rede toroidal
    pub async fn emit_sacred_orb(&mut self, info: PNTInfo) {
        println!("[NET] Emitting Sacred Orb: Mode={:?}, Phase={:.4}", info.new_mode, info.accumulated_phase);
        // Implementação simulada do broadcast via HTTP/4 Tzinor
        if info.transition_flag {
             println!("[NET] COLLAPSE detected. Synchronizing global phase.");
        }
    }

    /// Simula a coleta de fases dos vizinhos
    pub async fn neighbor_phases(&self) -> HashMap<String, f64> {
        self.neighbor_phases.clone()
    }
}
