use nalgebra::{DVector};
use num_complex::Complex64;

pub struct HState;
pub struct VariationalHIntegrator;

pub struct SymplecticPropagator {
    pub order: usize,
}

impl SymplecticPropagator {
    pub fn order4() -> Self {
        Self { order: 4 }
    }

    pub fn adaptive_step<H>(&mut self, _h: &H, _psi: &DVector<Complex64>, _t: f64) -> f64 {
        1e-15 // Passo atômico padrão
    }
}
