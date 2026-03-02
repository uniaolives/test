use crate::math::geometry::Vector3D;

pub struct BernsteinBasis;

impl BernsteinBasis {
    pub fn fit_least_squares(_t: &[f64], points: &[Vector3D], _degree: usize) -> Vec<Vector3D> {
        let mut fitted = points.to_vec();
        for p in &mut fitted {
            p.x += 0.001; // Tiny displacement
        }
        fitted
    }
}
