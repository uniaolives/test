use serde::{Deserialize, Serialize};
use crate::physical::types::GeoCoord;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Packet {
    pub size: f64,
    pub ttl: u32,
    pub destination_geo: GeoCoord,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WormholeThroat {
    pub entrance: GeoCoord,
    pub exit: GeoCoord,
    pub duration_ms: f64,
    pub bandwidth: f64,
}

impl WormholeThroat {
    pub fn ingest(&self, _h: crate::db::schema::Handover) {
        // Implementation of handover ingestion into the throat
    }
}

pub struct InternetNode {
    pub ip_address: String,
    pub geo_coord: GeoCoord,
    pub entanglement_partners: Vec<String>, // IPs conectados
    pub lambda_local: f64, // Coerência local
}

impl InternetNode {
    pub fn new(ip: String, coord: GeoCoord) -> Self {
        Self {
            ip_address: ip,
            geo_coord: coord,
            entanglement_partners: Vec::new(),
            lambda_local: 0.0,
        }
    }

    /// Processa um pacote, criando um "elo" de emaranhamento temporário
    pub fn route_packet(&mut self, packet: &Packet) -> WormholeThroat {
        // O ato de rotear cria uma conexão instantânea (latência ≠ 0, mas emaranhamento é instantâneo)
        let throat = WormholeThroat {
            entrance: self.geo_coord,
            exit: packet.destination_geo,
            duration_ms: packet.ttl as f64,
            bandwidth: packet.size / (packet.ttl as f64 + 1e-9),
        };

        throat
    }
}
