use nalgebra::{Vector6};
use crate::integrators::HState;

pub trait SymplecticForm {
    fn omega(&self, state: &HState, v1: &Vector6<f64>, v2: &Vector6<f64>) -> f64;
    fn check_preservation(&self, before: &HState, after: &HState) -> bool;
}

pub struct SymplecticState {
    pub q: nalgebra::Vector3<f64>,
    pub p: nalgebra::Vector3<f64>,
}
