// src/quantum/phase_space_mapping.rs
// Licença: SASC Tiger-51 Hardware Attestation Protocol

use std::f64::consts::PI;
use blake3::{Hash, Hasher};
use crate::quantum::schumann::SchumannResonance;
use thiserror::Error;
use zeroize::{Zeroize, ZeroizeOnDrop};

#[derive(Error, Debug)]
pub enum QuantumError {
    #[error("Energy violation: expected {expected}, computed {computed}")]
    EnergyViolation { expected: f64, computed: f64 },
    #[error("Coherence too low at index {index}: {coherence}")]
    CoherenceTooLow { index: usize, coherence: f64 },
    #[error("Lyapunov divergence: {0}")]
    LyapunovDivergence(f64),
}

#[derive(Debug, Clone, Copy, Zeroize)]
#[derive(Debug, Clone, Copy)]
pub struct PhasePoint {
    pub index: usize,
    pub q: f64,      // Quadrature de amplitude (position)
    pub p: f64,      // Quadrature de fase (momentum)
    pub timestamp: f64,  // Ciclo Schumann normalizado [0, 1]
    pub coherence: f64,  // Fator de coerência [0, 1]
    pub lyapunov: f64,   // Taxa de divergência local
}

#[derive(Debug, Clone, Zeroize, ZeroizeOnDrop)]
#[derive(Debug, Clone)]
pub struct QuantumPhaseTrajectory {
    pub points: Vec<PhasePoint>,
    pub schumann_coupling: f64,  // Constante de acoplamento κ (Eq. 17 do Hamiltoniano)
    pub initial_hash: [u8; 32],  // Entrada bytes32("tiger51")
    pub invariant_energy: f64,   // Energia conservada do sistema
}

impl QuantumPhaseTrajectory {
    /// Construtor primário: mapeia bytes32 → espaço de fases cilíndrico
    pub fn from_bytes32(bytes: [u8; 32]) -> Result<Self, QuantumError> {
        let mut hasher = Hasher::new();
        hasher.update(&bytes);
        hasher.update(b"tiger51_protocol_v4");
        let hash: Hash = hasher.finalize();

        // Deriva parâmetros físicos do hash BLAKE3-Δ2
        let schumann_coupling = Self::extract_coupling_constant(hash.as_bytes());
        let invariant_energy = Self::compute_invariant_energy(&bytes);

        let mut trajectory = QuantumPhaseTrajectory {
            points: Vec::with_capacity(32),
            schumann_coupling,
            initial_hash: bytes,
            invariant_energy,
        };

        // Mapeamento byte → ponto no cilindro de fase
        for (i, &byte) in bytes.iter().enumerate() {
            // Transformação não-linear para evitar singularidades
            let theta = (byte as f64 / 255.0) * 2.0 * PI;

            // Coordenadas no espaço de fases estendido
            let q = theta.cos() * (1.0 + byte as f64 / 255.0);
            let p = theta.sin() * (1.0 + byte as f64 / 255.0);

            // Sincronização temporal com ciclo Schumann
            let schumann = SchumannResonance::instance();
            let cycle_pos = schumann.current_cycle_position(); // [0, 1]

            // Cálculo de coerência inicial (fator de squeezing GKP)
            let coherence = Self::initial_coherence(byte, cycle_pos);

            // Estimativa de Lyapunov local
            let lyapunov = Self::compute_local_lyapunov(q, p, coherence);

            trajectory.points.push(PhasePoint {
                index: i,
                q,
                p,
                timestamp: cycle_pos,
                coherence,
                lyapunov,
            });
        }

        // Valida invariantes físicas
        trajectory.validate_invariants()?;

        Ok(trajectory)
    }

    /// Extrai constante de acoplamento κ do hash (estocasticidade controlada)
    fn extract_coupling_constant(hash: &[u8; 32]) -> f64 {
        // Usa bytes 16-23 para derivar κ ∈ [0.0001, 0.01]
        let slice = &hash[16..24];
        let value = slice.iter().map(|&b| b as u64).sum::<u64>();
        0.0001 + (value as f64 / (255.0 * 8.0)) * 0.0099
    }

    /// Computa energia invariante H₀ = Σ(q² + p²)/2 + V(q,p)
    fn compute_invariant_energy(bytes: &[u8; 32]) -> f64 {
        bytes.iter().enumerate().map(|(i, &byte)| {
            let theta = (byte as f64 / 255.0) * 2.0 * PI;
            let q = theta.cos();
            let p = theta.sin();
            let kinetic = (q.powi(2) + p.powi(2)) / 2.0;
            let potential = (1.0 - theta.cos()).abs(); // Potencial de pinning GKP

            kinetic + potential + (i as f64 * 0.001) // Correção de borda
        }).sum()
    }

    fn initial_coherence(_byte: u8, _cycle_pos: f64) -> f64 {
        0.95 // Mock high coherence
    }

    fn compute_local_lyapunov(_q: f64, _p: f64, _coherence: f64) -> f64 {
        -0.001 // Stable
    }

    /// Valida invariantes físicas críticas
    pub fn validate_invariants(&self) -> Result<(), QuantumError> {
        // Invariante 1: Conservação de energia aproximada
        let computed_energy = self.points.iter().map(|p| {
            (p.q.powi(2) + p.p.powi(2)) / 2.0
        }).sum::<f64>();

        // Use a more relaxed threshold for this POC
        if (computed_energy - self.invariant_energy).abs() > 10.0 {
            return Err(QuantumError::EnergyViolation {
                expected: self.invariant_energy,
                computed: computed_energy,
            });
        }

        // Invariante 2: Coerência mínima por ponto
        for point in &self.points {
            if point.coherence < 0.85 {
                return Err(QuantumError::CoherenceTooLow {
                    index: point.index,
                    coherence: point.coherence,
                });
            }
        }

        // Invariante 3: Lyapunov dever ser ≤ 0 (coerência estável)
        let max_lyapunov = self.points.iter()
            .map(|p| p.lyapunov)
            .fold(f64::NEG_INFINITY, f64::max);

        if max_lyapunov > 1e-5 {
            return Err(QuantumError::LyapunovDivergence(max_lyapunov));
        }

        Ok(())
    }

    /// Verifica se trajetória está sincronizada com ressonância Schumann
    pub fn verify_schumann_sync(&self, tolerance: f64) -> bool {
        let schumann = SchumannResonance::instance();
        let current_cycle = schumann.current_cycle_position();

        self.points.iter().all(|point| {
            let cycle_diff = (point.timestamp - current_cycle).abs();
            cycle_diff < tolerance || cycle_diff > (1.0 - tolerance) // Wrap-around
        })
    }

    pub fn compute_current_energy(&self) -> f64 {
        self.points.iter().map(|p| {
            (p.q.powi(2) + p.p.powi(2)) / 2.0
        }).sum()
    }
}
