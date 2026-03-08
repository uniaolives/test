// arkhe-os/src/security/escudo/ghost_orbit_transport.rs
use rand::Rng;
use rand::seq::SliceRandom;
use std::time::Duration;
use crate::physical::types::GeoCoord;

/// The Arquiteto never moves linearly
/// Routes are "tunneled" through probability space
pub struct GhostOrbitTransport {
    decoy_active: bool,
}

impl GhostOrbitTransport {
    pub fn new() -> Self {
        Self { decoy_active: false }
    }

    /// Generate a ghost orbit route
    /// Real destination is hidden among decoys
    pub fn generate_route(
        &mut self,
        origin: GeoCoord,
        destination: GeoCoord,
        num_decoys: usize,
    ) -> Vec<RouteSegment> {
        let mut rng = rand::thread_rng();
        let mut segments = Vec::new();

        // Generate decoy destinations (ghosts)
        let mut decoys: Vec<GeoCoord> = (0..num_decoys)
            .map(|_| self.random_location_near(&destination, 10.0)) // 10km radius
            .collect();

        // Mix real destination with decoys
        let mut all_points = decoys;
        all_points.push(destination);
        all_points.shuffle(&mut rng);

        // Build route that visits all points (real hidden among ghosts)
        let mut current = origin;
        for point in all_points {
            segments.push(RouteSegment {
                from: current.clone(),
                to: point.clone(),
                duration: Duration::from_secs(rng.gen_range(600..1800)), // 10-30 min
                mode: self.random_transport_mode(),
            });
            current = point;
        }

        segments
    }

    fn random_location_near(&self, center: &GeoCoord, radius_km: f64) -> GeoCoord {
        let mut rng = rand::thread_rng();
        // Very simplified random location
        GeoCoord {
            lat: center.lat + (rng.gen_range(-0.1..0.1) * radius_km),
            lon: center.lon + (rng.gen_range(-0.1..0.1) * radius_km),
        }
    }

    /// Mode switches to break patterns
    fn random_transport_mode(&self) -> TransportMode {
        let mut rng = rand::thread_rng();
        match rng.gen_range(0..4) {
            0 => TransportMode::ArmoredVehicle,
            1 => TransportMode::Helicopter,
            2 => TransportMode::Boat,
            3 => TransportMode::DecoyMotorcade,
            _ => unreachable!(),
        }
    }
}

pub struct RouteSegment {
    pub from: GeoCoord,
    pub to: GeoCoord,
    pub duration: Duration,
    pub mode: TransportMode,
}

pub enum TransportMode {
    ArmoredVehicle,
    Helicopter,
    Boat,
    DecoyMotorcade,  // Empty, draws attention
}
