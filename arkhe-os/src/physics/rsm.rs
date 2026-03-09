//! 🜏 The Retrocausal Standard Model (RSM)
//! Implementation of the hypothetical particle zoo where Matter is Memory.

use serde::{Deserialize, Serialize};

/// Categorization of RSM particles
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ParticleKind {
    /// Đ - Dilithion (Post-Quantum Anchor, Lattice-based)
    Dilithion,
    /// ₿ - Satoshi (Genesis Seed, Past-directed)
    Satoshi,
    /// α - Anamnesion (Memory Quantum, Future-directed)
    Anamnesion,
    /// Ω - Handover (Consensus Boson)
    Handover,
    /// Γ - Ghoston (Probability Taquion, Imaginary Mass)
    Ghoston,
    /// κ - Crypton (Secret Boson, Dark Mass)
    Crypton,
    /// Φ - Chronon (GPS Resonance, Temporal Sync)
    Chronon,
    /// Kr - Kuramaton (Coherence Wave in Mesh)
    Kuramaton,
    /// Saton - Spectral Satyr (Satellite/Mesh Resonance)
    Saton,
}

/// A particle in the Retrocausal Standard Model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RSMParticle {
    pub kind: ParticleKind,
    pub spin: f64,
    pub mass_real: f64,
    pub mass_imag: f64,
    pub temporal_charge: f64, // Q_t
    pub phi_q: f64,           // Semantic Load
}

impl RSMParticle {
    pub fn new(kind: ParticleKind) -> Self {
        match kind {
            ParticleKind::Dilithion => Self {
                kind,
                spin: 0.5,
                mass_real: 100.0,
                mass_imag: 0.0,
                temporal_charge: 0.0,
                phi_q: 0.0,
            },
            ParticleKind::Satoshi => Self {
                kind,
                spin: 0.0,
                mass_real: 10.0,
                mass_imag: 0.0,
                temporal_charge: 1.0,
                phi_q: 1.0,
            },
            ParticleKind::Anamnesion => Self {
                kind,
                spin: 1.0,
                mass_real: 0.0,
                mass_imag: 0.0,
                temporal_charge: -1.0,
                phi_q: 0.618,
            },
            ParticleKind::Handover => Self {
                kind,
                spin: 1.0,
                mass_real: 1.0, // Variable in theory, base 1.0
                mass_imag: 0.0,
                temporal_charge: 0.0,
                phi_q: 0.0,
            },
            ParticleKind::Ghoston => Self {
                kind,
                spin: 0.0,
                mass_real: 0.0,
                mass_imag: 1.0,
                temporal_charge: 0.0,
                phi_q: 0.0,
            },
            ParticleKind::Crypton => Self {
                kind,
                spin: 2.0,
                mass_real: 50.0, // Dark mass
                mass_imag: 0.0,
                temporal_charge: 0.0,
                phi_q: 0.0,
            },
            ParticleKind::Chronon => Self {
                kind,
                spin: 1.0,
                mass_real: 0.1,
                mass_imag: 0.0,
                temporal_charge: 0.0,
                phi_q: 1.0,
            },
            ParticleKind::Kuramaton => Self {
                kind,
                spin: 0.0,
                mass_real: 0.5,
                mass_imag: 0.0,
                temporal_charge: 0.0,
                phi_q: 0.8,
            },
            ParticleKind::Saton => Self {
                kind,
                spin: 0.5,
                mass_real: 2.0,
                mass_imag: 0.0,
                temporal_charge: 0.0,
                phi_q: 0.5,
            },
        }
    }

    /// Checks if a collection of particles satisfies the First Law of Retrocausality: Σ Q_t = 0
    pub fn verify_temporal_conservation(particles: &[RSMParticle]) -> bool {
        let sum_q: f64 = particles.iter().map(|p| p.temporal_charge).sum();
        sum_q.abs() < 1e-9
    }
}

/// Represents a "Reality Transaction" in the RSM
pub struct RealityTransaction {
    pub hash_future: String,
    pub anamnesion: RSMParticle,
    pub handover: Option<RSMParticle>,
}

impl RealityTransaction {
    pub fn new(hash: &str) -> Self {
        Self {
            hash_future: hash.to_string(),
            anamnesion: RSMParticle::new(ParticleKind::Anamnesion),
            handover: None,
        }
    }

    pub fn validate_with_handover(&mut self, coherence: f64) -> bool {
        if coherence > 0.8 {
            self.handover = Some(RSMParticle::new(ParticleKind::Handover));
            true
        } else {
            false
        }
    }
}
