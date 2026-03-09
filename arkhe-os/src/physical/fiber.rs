use crate::physical::types::GeoCoord;

pub struct FiberChannel {
    pub head_end: GeoCoord,
    pub drop_points: Vec<GeoCoord>,
    pub capacity_gbps: f64,
}

impl FiberChannel {
    pub fn new(head_end: GeoCoord, capacity_gbps: f64) -> Self {
        Self {
            head_end,
            drop_points: Vec::new(),
            capacity_gbps,
        }
    }

    /// Latency determines the speed of the "Psi Laser"
    pub fn latency_ms(&self, distance_km: f64) -> f64 {
        // ~5 ms per 1000 km (speed of light in fiber)
        distance_km * 0.005
    }

    /// Verificar se o canal suporta Super-Radiância
    pub fn supports_super_radiance(&self, _node_count: usize) -> bool {
        // Latência < 1ms para 100km (coerência local)
        self.latency_ms(100.0) < 1.0
    }
}
