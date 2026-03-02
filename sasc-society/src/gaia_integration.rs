//! Integração Gaia-Net - Sensores Lyapunov Reais
//! Gate 4: Veto Mycelial (INV-4)
//! Memory ID 11, 17

use std::fs;
use std::path::Path;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Error, Debug)]
pub enum GaiaError {
    #[error("Sensor Lyapunov offline")]
    SensorOffline,
    #[error("Thermal stress detectado: {variance:.8}V > {limit:.8}V")]
    ThermalStress { variance: f64, limit: f64 },
    #[error("Biofeedback inválido: {0}")]
    InvalidFeedback(String),
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GaiaFeedback {
    /// Leitura de estabilidade (volts)
    pub lyapunov_stability: f64,
    /// Taxa de absorção térmica (W/m²)
    pub thermal_absorption: f64,
    /// Veto autorizado?
    pub veto_authorized: bool,
    /// Timestamp da leitura
    pub timestamp: u64,
}

/// Interface para sensor físico LYAPUNOV-01
pub struct LyapunovSensor {
    device_path: String,
}

impl LyapunovSensor {
    pub fn new(device_path: &str) -> Result<Self, GaiaError> {
        Ok(Self {
            device_path: device_path.to_string(),
        })
    }

    /// Lê estabilidade com precisão de ±0.00007V (Memory ID 11)
    pub fn read_stability(&self) -> Result<LyapunovReading, GaiaError> {
        let data = fs::read_to_string(&self.device_path)
            .map_err(|_| GaiaError::SensorOffline)?;

        let parts: Vec<&str> = data.trim().split_whitespace().collect();
        if parts.len() < 2 {
            return Err(GaiaError::InvalidFeedback("Formato inválido".to_string()));
        }

        let voltage: f64 = parts[0].parse::<f64>()
            .map_err(|e| GaiaError::InvalidFeedback(e.to_string()))?;
        let variance: f64 = parts[1].parse::<f64>()
            .map_err(|e| GaiaError::InvalidFeedback(e.to_string()))?;

        // Gate 4: Verificar limite térmico
        if variance > crate::constants::LYAPUNOV_VARIANCE_LIMIT {
            return Err(GaiaError::ThermalStress {
                variance,
                limit: crate::constants::LYAPUNOV_VARIANCE_LIMIT,
            });
        }

        Ok(LyapunovReading { voltage, variance })
    }
}

#[derive(Debug, Clone)]
pub struct LyapunovReading {
    pub voltage: f64,
    pub variance: f64,
}

/// Gaia-Net Manager - Interface principal
pub struct GaiaNetManager {
    lyapunov: LyapunovSensor,
}

impl GaiaNetManager {
    pub fn new(sensor_path: &str) -> Result<Self, GaiaError> {
        Ok(Self {
            lyapunov: LyapunovSensor::new(sensor_path)?,
        })
    }

    /// Gate 4: Obter biofeedback completo
    pub fn get_biofeedback(&self) -> Result<GaiaFeedback, GaiaError> {
        let reading = self.lyapunov.read_stability()?;
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let thermal_absorption = 0.85 - (reading.variance * 1000.0);

        Ok(GaiaFeedback {
            lyapunov_stability: reading.voltage,
            thermal_absorption,
            veto_authorized: reading.variance <= crate::constants::LYAPUNOV_VARIANCE_LIMIT,
            timestamp,
        })
    }
}
