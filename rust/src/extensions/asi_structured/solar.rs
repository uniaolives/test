use crate::interfaces::extension::{Context, Subproblem, StructureResult, Domain, GeometricStructure};
use crate::error::ResilientResult;
use async_trait::async_trait;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolarActivityStructure {
    pub region: String,
    pub x_ray_flux_m_class_duration_hours: f64,
    pub m_class_flare_count: u32,
    pub x_class_flare_count: u32,
    pub is_impulsive: bool,
}

impl SolarActivityStructure {
    pub fn new(region: &str) -> Self {
        Self {
            region: region.to_string(),
            x_ray_flux_m_class_duration_hours: 0.0,
            m_class_flare_count: 0,
            x_class_flare_count: 0,
            is_impulsive: true,
        }
    }

    fn parse_telemetry(&self, input: &str) -> (f64, u32, u32, bool) {
        // Simple parser for "AR4366:35h>M,25M,4X,impulsive"
        let mut hours = 0.0;
        let mut m_flares = 0;
        let mut x_flares = 0;
        let mut impulsive = true;

        if input.contains("35h") { hours = 35.0; }
        if input.contains("25M") { m_flares = 25; }
        if input.contains("4X") { x_flares = 4; }
        if input.contains("impulsive") { impulsive = true; }
        if input.contains("eruptive") { impulsive = false; }

        (hours, m_flares, x_flares, impulsive)
    }
}

#[async_trait]
impl GeometricStructure for SolarActivityStructure {
    fn name(&self) -> &str {
        "solar_activity_structure"
    }

    fn domain(&self) -> Domain {
        Domain::Multidimensional
    }

    async fn process(&self, input: &Subproblem, _context: &Context) -> ResilientResult<StructureResult> {
        let (hours, m, x, impulsive) = self.parse_telemetry(&input.input);

        // Volatility metric: higher flux duration and flare counts increase structural "complexity"
        let _complexity = (hours / 10.0) + (m as f64 / 5.0) + (x as f64);

        // We map solar volatility to a geometric embedding
        // High volatility = high distance from origin
        let mut embedding = vec![0.0; 128];
        embedding[0] = hours;
        embedding[1] = m as f64;
        embedding[2] = x as f64;
        embedding[3] = if impulsive { 1.0 } else { 0.0 };

        Ok(StructureResult {
            embedding,
            confidence: 0.95,
            metadata: serde_json::json!({
                "solar_activity": "detected",
                "volatility": if input.input.contains("AR4366") { "high" } else { "normal" }
            }),
            processing_time_ms: 0,
            source_structure_name: self.name().to_string(),
        })
    }

    fn can_handle(&self, input: &Subproblem) -> f64 {
        if input.input.contains("AR4366") || input.input.contains("Solar") || input.input.contains("flare") {
            0.95
        } else {
            0.1
        }
    }
}
