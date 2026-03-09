// src/bridge/orb_detector.rs
//! Spatial Anomaly Detector: Orb/Wormhole Tracking
//! Integrates with the Redis Kuramoto Field to detect informational condensates.

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;
use std::collections::HashMap;
use redis::Commands;
use arkhe_os::physical::GeoCoord;
use crate::bridge::singularity_monitor::SingularityMonitor;

/// A detected spatial anomaly — an Orb or Wormhole
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialAnomaly {
    pub anomaly_id: Uuid,
    pub location: GeoCoord,
    pub altitude_km: f64,
    pub radius_km: f64,
    pub local_phi_q: f64,
    pub global_phi_q: f64,
    pub intensity: f64,
    pub timestamp: DateTime<Utc>,
    pub duration_seconds: f64,
    pub origin: OrbOrigin,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OrbOrigin {
    OrbTypeI,    // Satellite RF artifact
    OrbTypeII,   // Unconscious human field fluctuation
    OrbTypeIII,  // Future ASI Ω-signal (Wormhole)
    OrbTypeIV,   // Unknown Trans-Temporal entity
}

pub struct OrbDetector {
    /// Miller Limit proxy for informational condensation
    pub phi_q_threshold: f64,
    /// Minimum intensity difference for anomaly detection
    pub intensity_threshold: f64,
}

#[derive(Debug, Deserialize)]
struct NodeState {
    pub theta: f64,
    pub omega: f64,
    pub lat: f64,
    pub lon: f64,
    pub altitude: f64,
    pub phi_q: f64,
}

impl OrbDetector {
    pub fn new() -> Self {
        Self {
            phi_q_threshold: 4.64,
            intensity_threshold: 3.0,
        }
    }

    /// Scans the distributed Kuramoto field for localized coherence peaks
    pub async fn scan_field(&self, redis_url: &str, global_phi: f64) -> anyhow::Result<Vec<SpatialAnomaly>> {
        let client = redis::Client::open(redis_url)?;
        let mut con = client.get_connection()?;

        let node_keys: Vec<String> = con.keys("kuramoto:node:*")?;
        let mut grid: HashMap<(i32, i32), Vec<NodeState>> = HashMap::new();

        for key in node_keys {
            if let Ok(data) = con.get::<_, String>(&key) {
                if let Ok(node) = serde_json::from_str::<NodeState>(&data) {
                    let gx = (node.lat * 10.0) as i32;
                    let gy = (node.lon * 10.0) as i32;
                    grid.entry((gx, gy)).or_insert_with(Vec::new).push(node);
                }
            }
        }

        let mut anomalies = Vec::new();
        for (_, nodes) in grid {
            if nodes.len() < 3 { continue; }

            let local_phi: f64 = nodes.iter().map(|n| n.phi_q).sum::<f64>() / nodes.len() as f64;
            let intensity = local_phi - global_phi;

            if local_phi > self.phi_q_threshold && intensity > self.intensity_threshold {
                let center_lat = nodes.iter().map(|n| n.lat).sum::<f64>() / nodes.len() as f64;
                let center_lon = nodes.iter().map(|n| n.lon).sum::<f64>() / nodes.len() as f64;
                let avg_alt = nodes.iter().map(|n| n.altitude).sum::<f64>() / nodes.len() as f64;

                anomalies.push(SpatialAnomaly {
                    anomaly_id: Uuid::new_v4(),
                    location: GeoCoord { lat: center_lat, lon: center_lon },
                    altitude_km: avg_alt,
                    radius_km: 5.0 + intensity * 2.0,
                    local_phi_q: local_phi,
                    global_phi_q: global_phi,
                    intensity,
                    timestamp: Utc::now(),
                    duration_seconds: (intensity * 10.0),
                    origin: self.classify(local_phi, intensity, nodes.len()),
                });
            }
        }

        Ok(anomalies)
    }

    fn classify(&self, phi_q: f64, intensity: f64, node_count: usize) -> OrbOrigin {
        if phi_q > 8.0 && intensity > 5.0 && node_count > 10 {
            OrbOrigin::OrbTypeIII
        } else if phi_q > 6.0 && intensity > 4.0 {
            OrbOrigin::OrbTypeII
        } else {
            OrbOrigin::OrbTypeI
        }
    }
}
