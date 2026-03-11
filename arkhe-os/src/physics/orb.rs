use serde::{Deserialize, Serialize};
use crate::physics::internet_mesh::WormholeThroat;
use crate::db::schema::Handover;
use thiserror::Error;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum RFSource {
    Satellite,
    Terrestrial,
    DeepSpace,
    HydrogenBase, // 1420.4556 MHz
}

#[derive(Error, Debug)]
pub enum WormholeCollapse {
    #[error("Decoherence detected")]
    Decoherence,
    #[error("Energy surge")]
    EnergySurge,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Orb {
    pub throat_geometry: WormholeThroat,
    pub stability: f64,      // λ₂ da região
    pub energy_source: RFSource,
    pub oam_topology_l: Option<i32>, // Orbital Angular Momentum
}

impl Orb {
    /// Abre um canal de comunicação através do Orb
    pub fn transmit_handover(&self, h: Handover) -> Result<(), WormholeCollapse> {
        // Injeta o handover na garganta
        self.throat_geometry.ingest(h);

        // Verifica estabilidade
        if self.stability < 0.5 {
            return Err(WormholeCollapse::Decoherence);
        }

        Ok(())
    }
}
