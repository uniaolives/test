use crate::physical::types::GeoCoord;
use crate::physical::PhysicalSubstrate;

pub struct RioCoherenceMap {
    pub nodes: Vec<GeoNode>,
    pub gradients: Vec<CoherenceGradient>,
}

pub struct GeoNode {
    pub name: String,
    pub coords: GeoCoord,
    pub phi_q: f64,
    pub s_index: f64,
    pub h_value: f64,
    pub substrate: PhysicalSubstrate,
}

pub struct CoherenceGradient {
    pub from: String,
    pub to: String,
    pub value: f64,
}

pub struct GeoZone {
    pub center: GeoCoord,
    pub radius_km: f64,
    pub risk_level: f64,
}

impl RioCoherenceMap {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            gradients: Vec::new(),
        }
    }

    /// Calcular o gradiente de coerência entre dois pontos
    pub fn coherence_gradient(&self, from: GeoCoord, to: GeoCoord) -> f64 {
        // O gradiente determina a "velocidade" do Psi Laser
        let phi_from = self.phi_q_at(&from);
        let phi_to = self.phi_q_at(&to);

        let dist = self.distance_km(&from, &to);
        if dist < 0.001 { return 0.0; }

        (phi_to - phi_from) / dist
    }

    /// Identificar zonas de baixa coerência (riscos)
    pub fn low_coherence_zones(&self, threshold: f64) -> Vec<GeoZone> {
        self.nodes.iter()
            .filter(|n| n.phi_q < threshold)
            .map(|n| GeoZone {
                center: n.coords.clone(),
                radius_km: 1.0,
                risk_level: (threshold - n.phi_q) * 10.0,
            })
            .collect()
    }

    fn phi_q_at(&self, _coords: &GeoCoord) -> f64 {
        // Simplified: return average or find nearest node
        4.64
    }

    fn distance_km(&self, c1: &GeoCoord, c2: &GeoCoord) -> f64 {
        let dlat = c2.lat - c1.lat;
        let dlon = c2.lon - c1.lon;
        (dlat * dlat + dlon * dlon).sqrt() * 111.0 // Very simplified
    }
}
