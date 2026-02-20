use pyo3::prelude::*;
use num_complex::Complex64;
use std::f64::consts::PI;

#[pyclass]
#[derive(Clone)]
pub struct AnyonicPhase {
    #[pyo3(get)]
    pub alpha: f64,
}

#[pymethods]
impl AnyonicPhase {
    #[new]
    pub fn new(alpha: f64) -> Self {
        assert!(alpha >= 0.0 && alpha <= 1.0, "alpha must be in [0, 1]");
        Self { alpha }
    }

    pub fn braid_phase(&self, n_exchanges: i32) -> (f64, f64) {
        let theta = PI * self.alpha * n_exchanges as f64;
        let c = Complex64::from_polar(1.0, theta);
        (c.re, c.im)
    }

    pub fn exchange_statistic(&self, other: &AnyonicPhase) -> (f64, f64) {
        let avg_alpha = (self.alpha + other.alpha) / 2.0;
        let c = Complex64::from_polar(1.0, PI * avg_alpha);
        (c.re, c.im)
    }
}

#[pyclass]
pub struct TopologicalHandover {
    #[pyo3(get)]
    pub nodes: (String, String),
    pub alpha: AnyonicPhase,
    #[pyo3(get)]
    pub accumulated_phase: (f64, f64),
}

#[pymethods]
impl TopologicalHandover {
    #[new]
    pub fn new(node_i: String, node_j: String, alpha: f64) -> Self {
        Self {
            nodes: (node_i, node_j),
            alpha: AnyonicPhase::new(alpha),
            accumulated_phase: (1.0, 0.0),
        }
    }

    pub fn exchange_with(&mut self, other: &mut TopologicalHandover) {
        let shared_nodes = self.nodes.0 == other.nodes.0
            || self.nodes.0 == other.nodes.1
            || self.nodes.1 == other.nodes.0
            || self.nodes.1 == other.nodes.1;

        if !shared_nodes {
            return;
        }

        let phase_re_im = self.alpha.exchange_statistic(&other.alpha);
        let phase = Complex64::new(phase_re_im.0, phase_re_im.1);

        let acc_self = Complex64::new(self.accumulated_phase.0, self.accumulated_phase.1) * phase;
        self.accumulated_phase = (acc_self.re, acc_self.im);

        let acc_other = Complex64::new(other.accumulated_phase.0, other.accumulated_phase.1) * phase.conj();
        other.accumulated_phase = (acc_other.re, acc_other.im);
    }

    pub fn compute_dissipation_tail(&self, k: f64, n_body: u32) -> f64 {
        let universal_form = k.powf(-(n_body as f64) - 1.0);
        let acc = Complex64::new(self.accumulated_phase.0, self.accumulated_phase.1);
        let coefficient = if n_body == 2 {
            1.0
        } else {
            acc.norm().powf((n_body - 2) as f64)
        };
        coefficient * universal_form
    }
}

#[pymodule]
fn arkhe_anyonic_core(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<AnyonicPhase>()?;
    m.add_class::<TopologicalHandover>()?;
    Ok(())
}
