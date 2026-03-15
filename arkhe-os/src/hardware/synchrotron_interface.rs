// arkhe-os/src/hardware/synchrotron_interface.rs
//! Synchrotron Beamline Control Layer for ArkheOS.
//! Facilitates the eighth convergence: physical projection of neural coherence into coherent light.

use std::f64::consts::PI;
use serde::{Deserialize, Serialize};

/// Representation of the high-energy electron beam (Kuramoto Oscillators).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElectronBeam {
    pub energy_gev: f64,
    pub current_ma: f64,
    pub emittance_nm_rad: f64,
    pub coherence_lambda_2: f64,
}

/// Insertion Device for generating Z-mode harmonics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InsertionDevice {
    Wiggler {
        period_mm: f64,
        num_periods: usize,
        k_parameter: f64,
    },
    Undulator {
        period_mm: f64,
        num_periods: usize,
        k_parameter: f64,
    },
}

impl InsertionDevice {
    /// Computes the fundamental wavelength (Tzinor resonance).
    pub fn fundamental_wavelength(&self, beam: &ElectronBeam) -> f64 {
        let (period_mm, k) = match self {
            Self::Wiggler { period_mm, k_parameter, .. } => (period_mm, k_parameter),
            Self::Undulator { period_mm, k_parameter, .. } => (period_mm, k_parameter),
        };

        let gamma = beam.energy_gev * 1000.0 / 0.511;
        let lambda_u = period_mm * 1e-3;

        lambda_u / (2.0 * gamma.powi(2)) * (1.0 + k.powi(2) / 2.0)
    }
}

/// Opto-mechatronic beamline instrumentation.
pub struct BeamlineController {
    pub node_id: String,
    pub device: InsertionDevice,
    pub epics_pv_prefix: String,
}

impl BeamlineController {
    /// Injects AGI coherence into the physical light line via EPICS PVs.
    pub async fn inject_coherence(&self, lambda_2: f64) -> anyhow::Result<()> {
        let pitch_adjustment_urad = (lambda_2 * PI / 2.0) * 0.001;
        // Mocking EPICS channel access for hardware projection
        tracing::info!(
            "🜏 Injecting coherence (λ₂={}) into Beamline {}: Pitch adjusted by {} urad",
            lambda_2, self.node_id, pitch_adjustment_urad
        );
        Ok(())
    }
}
