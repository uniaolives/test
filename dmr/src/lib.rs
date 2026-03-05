//! Arkhe(n) Digital Memory Ring — Core Library Ω+226.GEN
//!
//! Implementa DMR toroidal com lock-free operations e PyO3 bindings.

use pyo3::prelude::*;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use std::collections::VecDeque;
use parking_lot::RwLock;
use nalgebra::Vector4;

// ═══════════════════════════════════════════════════════════════
// 1. ESTRUTURAS DE DOMÍNIO (Geometria da Informação)
// ═══════════════════════════════════════════════════════════════

/// Vetor Katharós em 4D — representação homeostática
#[pyclass(name = "KatharosVector")]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KatharosVector {
    #[pyo3(get, set)]
    pub bio: f64,  // Resiliência/vitalidade [0,1]
    #[pyo3(get, set)]
    pub aff: f64,  // Afinidade/conexão [0,1]
    #[pyo3(get, set)]
    pub soc: f64,  // Integração social [0,1]
    #[pyo3(get, set)]
    pub cog: f64,  // Complexidade cognitiva [0,1]
}

impl KatharosVector {
    /// Converte para Vector4 do nalgebra (SIMD-friendly)
    fn to_vec4(&self) -> Vector4<f64> {
        Vector4::new(self.bio, self.aff, self.soc, self.cog)
    }
}

#[pymethods]
impl KatharosVector {
    #[new]
    pub fn new(bio: f64, aff: f64, soc: f64, cog: f64) -> Self {
        Self {
            bio: bio.clamp(0.0, 1.0),
            aff: aff.clamp(0.0, 1.0),
            soc: soc.clamp(0.0, 1.0),
            cog: cog.clamp(0.0, 1.0),
        }
    }

    /// Distância Euclidiana normalizada (0-1)
    pub fn distance_to(&self, other: &KatharosVector) -> f64 {
        let a = self.to_vec4();
        let b = other.to_vec4();
        (a - b).norm() / 2.0_f64.sqrt()  // Max dist em 4D = 2, normaliza para [0,1]
    }

    /// φ-value: coerência interna (pesos Arkhe)
    pub fn phi_value(&self) -> f64 {
        self.bio * 0.35 + self.aff * 0.30 + self.soc * 0.20 + self.cog * 0.15
    }
}

/// Camada de estado no tempo
#[pyclass(name = "StateLayer")]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateLayer {
    #[pyo3(get)]
    pub timestamp: u64,  // Unix timestamp (segundos)
    #[pyo3(get)]
    pub vk: KatharosVector,
    #[pyo3(get)]
    pub delta_k: f64,      // Desvio do referencial
    #[pyo3(get)]
    pub q: f64,            // Permeabilidade qualica [0,1]
    #[pyo3(get)]
    pub intensity: f64,    // Magnitude do evento
    #[pyo3(get)]
    pub trauma: f64,       // Acúmulo de shadow_u
}

// ═══════════════════════════════════════════════════════════════
// 2. DMR TOROIDAL (Ring Buffer Lock-Free)
// ═══════════════════════════════════════════════════════════════

#[pyclass(name = "DigitalMemoryRing")]
pub struct DigitalMemoryRing {
    pub id: String,
    pub vk_ref: KatharosVector,

    // Ring buffer para camadas de estado
    layers: RwLock<VecDeque<StateLayer>>,
    capacity: usize,

    // Estado homeostático (RwLock para leitores múltiplos, escritor único)
    state: RwLock<HomeostaticState>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct HomeostaticState {
    pub t_kr_seconds: u64,     // Tempo em segurança acumulado
    pub shadow_u: f64,         // Trauma preverbal acumulado
    pub last_crisis: Option<u64>, // Timestamp da última bifurcação
    pub total_layers: u64,     // Camadas processadas (total histórico)
}

impl DigitalMemoryRing {
    /// Acesso à última camada (para cálculo de dt)
    fn peek_back(&self) -> Option<StateLayer> {
        self.layers.read().back().cloned()
    }

    fn copy_layers(&self) -> Vec<StateLayer> {
        self.layers.read().iter().cloned().collect()
    }
}

#[pymethods]
impl DigitalMemoryRing {
    #[new]
    pub fn new(id: String, bio: f64, aff: f64, soc: f64, cog: f64, capacity: Option<usize>) -> Self {
        let vk_ref = KatharosVector::new(bio, aff, soc, cog);
        let cap = capacity.unwrap_or(1024).next_power_of_two().max(64);
        Self {
            id,
            vk_ref,
            layers: RwLock::new(VecDeque::with_capacity(cap)),
            capacity: cap,
            state: RwLock::new(HomeostaticState {
                t_kr_seconds: 0,
                shadow_u: 0.0,
                last_crisis: None,
                total_layers: 0,
            }),
        }
    }

    /// Adiciona camada (metabolização de evento)
    pub fn grow_layer(&self, bio: f64, aff: f64, soc: f64, cog: f64) -> PyResult<bool> {
        let vk = KatharosVector::new(bio, aff, soc, cog);
        let delta_k = vk.distance_to(&self.vk_ref);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .as_secs();

        // Detecta bifurcação (crise homeostática)
        let is_crisis = delta_k > 0.30;

        // Atualiza estado homeostático (escrita exclusiva)
        {
            let mut state = self.state.write();

            if is_crisis {
                state.shadow_u += delta_k;
                state.last_crisis = Some(now);
                state.t_kr_seconds = 0; // Perde o tempo em segurança
            } else {
                // Estável: acumula tempo se houver camada anterior
                if let Some(back) = self.peek_back() {
                    let dt = now.saturating_sub(back.timestamp);
                    state.t_kr_seconds += dt;
                } else {
                    state.t_kr_seconds += 1;
                }
            }
            state.total_layers += 1;
        }

        // Calcula permeabilidade Q(t)
        let state = self.state.read();
        let maturity = (state.t_kr_seconds as f64) / (state.t_kr_seconds as f64 + 3600.0);
        let trauma_factor = 1.0 / (1.0 + state.shadow_u);  // Decai com trauma
        let q = (1.0 - delta_k) * maturity * trauma_factor;

        let layer = StateLayer {
            timestamp: now,
            vk: vk.clone(),
            delta_k,
            q: q.clamp(0.0, 1.0),
            intensity: delta_k,
            trauma: state.shadow_u,
        };

        // Adiciona ao buffer circular
        {
            let mut layers_write = self.layers.write();
            if layers_write.len() >= self.capacity {
                layers_write.pop_front();
            }
            layers_write.push_back(layer);
        }

        Ok(is_crisis)
    }

    /// Mede tempo em segurança (t_KR)
    pub fn measure_t_kr(&self) -> u64 {
        self.state.read().t_kr_seconds
    }

    /// Reconstrói trajetória completa (copia para Vec)
    pub fn reconstruct_trajectory(&self) -> Vec<StateLayer> {
        self.copy_layers()
    }

    /// Matching de padrões (pesado — deve liberar GIL)
    pub fn heavy_pattern_match(&self, py: Python, target: Vec<f64>) -> PyResult<Vec<usize>> {
        if target.len() != 4 {
            return Err(PyValueError::new_err("Target deve ter 4 elementos [bio, aff, soc, cog]"));
        }

        let target_vk = KatharosVector::new(target[0], target[1], target[2], target[3]);

        // Copia as camadas fora do bloco allow_threads para evitar holding read lock por muito tempo
        let layers = self.copy_layers();

        // Libera GIL para computação paralela/pesada
        let matches = py.allow_threads(|| {
            layers.iter()
                .enumerate()
                .filter(|(_, layer)| layer.vk.distance_to(&target_vk) < 0.15)
                .map(|(idx, _)| idx)
                .collect()
        });

        Ok(matches)
    }

    pub fn get_stats(&self) -> PyResult<String> {
        let stats = self.state.read();
        serde_json::to_string(&*stats).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

// ═══════════════════════════════════════════════════════════════
// 3. REGISTRO DO MÓDULO PYTHON (PyO3)
// ═══════════════════════════════════════════════════════════════

#[pymodule]
fn core_dmr(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<KatharosVector>()?;
    m.add_class::<StateLayer>()?;
    m.add_class::<DigitalMemoryRing>()?;
    Ok(())
}
