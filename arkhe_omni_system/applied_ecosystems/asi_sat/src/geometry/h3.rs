// arkhe_omni_system/applied_ecosystems/asi_sat/src/geometry/h3.rs
use nalgebra::Vector3;

#[derive(Debug, Clone, Copy)]
pub struct H3Point {
    pub x: f64,
    pub y: f64,
    pub z: f64, // z > 0 (altitude/scale)
}

impl H3Point {
    /// Compute hyperbolic distance in the upper half-space model:
    /// d_H(p1, p2) = acosh(1 + ||p1 - p2||^2 / (2 * z1 * z2))
    pub fn dist_to(&self, other: &H3Point) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        let sq_dist = dx*dx + dy*dy + dz*dz;
        let arg = 1.0 + sq_dist / (2.0 * self.z * other.z);
        arg.acosh()
    }

    /// Map spherical LEO coordinates to H3
    /// lon: longitude (rad), lat: latitude (rad), alt: altitude (km)
    pub fn from_orbital(lon: f64, lat: f64, alt: f64) -> Self {
        // Projection onto boundary plane z=0
        // Scale height H0 ensures z > 0
        let r_earth = 6371.0;
        let x = lon * r_earth;
        let y = lat * r_earth;
        let z = alt + 100.0; // Offset for stability
        Self { x, y, z }
    }
}

pub struct GreedyRouter;

impl GreedyRouter {
    /// Find the next hop neighbor that minimizes hyperbolic distance to target
    pub fn next_hop(current: &H3Point, neighbors: &[H3Point], target: &H3Point) -> Option<usize> {
        let mut best_idx = None;
        let mut min_dist = current.dist_to(target);

        for (i, neighbor) in neighbors.iter().enumerate() {
            let dist = neighbor.dist_to(target);
            if dist < min_dist {
                min_dist = dist;
                best_idx = Some(i);
            }
        }
        best_idx
    }
}
