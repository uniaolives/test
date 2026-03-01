// arkhe_omni_system/applied_ecosystems/asi_sat/src/geometry/poincare.rs
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

#[derive(Debug, Clone, Copy)]
pub struct PoincarePoint {
    pub r: Decimal,     // Radial distance (0 <= r < 1)
    pub theta: Decimal, // Azimuthal angle (longitude)
    pub zeta: Decimal,  // Polar projection (latitude sin)
}

impl PoincarePoint {
    /// Embedding algorithm: Geographic (Lat/Long/Alt) -> ℍ³ Poincare Ball
    /// lat: latitude (degrees), lon: longitude (degrees), alt: altitude (meters)
    pub fn from_geographic(lat: f64, lon: f64, alt: f64) -> Self {
        let r_earth = 6371000.0;

        // altitude defines the 'depth' into the ball
        // We use tanh for mapping to [0, 1)
        let r_val = (alt / (alt + r_earth)).tanh();

        let theta_val = lon.to_radians();
        let zeta_val = lat.to_radians().sin();

        // Convert to Decimal for precision compliance
        PoincarePoint {
            r: Decimal::from_f64_retain(r_val).unwrap_or(dec!(0)),
            theta: Decimal::from_f64_retain(theta_val).unwrap_or(dec!(0)),
            zeta: Decimal::from_f64_retain(zeta_val).unwrap_or(dec!(0)),
        }
    }
}
