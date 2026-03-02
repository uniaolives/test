use nalgebra::DMatrix;
use num_complex::Complex64;
use rand::Rng;

/// Parâmetros de um operador de Kraus (matriz complexa de dimensão dim).
#[derive(Clone)]
pub struct KrausParams {
    pub real: Vec<f64>,
    pub imag: Vec<f64>,
    pub dim: usize,
}

impl KrausParams {
    pub fn random(dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let n = dim * dim;
        let real = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let imag = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
        KrausParams { real, imag, dim }
    }

    pub fn to_matrix(&self) -> DMatrix<Complex64> {
        let mut mat = DMatrix::from_element(self.dim, self.dim, Complex64::new(0.0, 0.0));
        for i in 0..self.dim {
            for j in 0..self.dim {
                let idx = i * self.dim + j;
                mat[(i, j)] = Complex64::new(self.real[idx], self.imag[idx]);
            }
        }
        mat
    }

    pub fn update(&mut self, grad: &Self, learning_rate: f64) {
        for i in 0..self.real.len() {
            self.real[i] -= learning_rate * grad.real[i];
            self.imag[i] -= learning_rate * grad.imag[i];
        }
    }
}

pub fn free_energy_for_kraus(
    k: &DMatrix<Complex64>,
    rho: &DMatrix<Complex64>,
    target: &DMatrix<Complex64>,
) -> f64 {
    let rho_new = k * rho * k.adjoint();
    let trace = rho_new.trace().re;
    if trace <= 1e-12 {
        return 1e10;
    }
    let rho_new_norm = rho_new / Complex64::new(trace, 0.0);
    (rho_new_norm - target).norm()
}

pub fn numerical_gradient(
    params: &KrausParams,
    rho: &DMatrix<Complex64>,
    target: &DMatrix<Complex64>,
    epsilon: f64,
) -> KrausParams {
    let dim = params.dim;
    let n = dim * dim;
    let mut grad_real = vec![0.0; n];
    let mut grad_imag = vec![0.0; n];

    let base_f = free_energy_for_kraus(&params.to_matrix(), rho, target);

    for idx in 0..n {
        let mut params_plus = params.clone();
        params_plus.real[idx] += epsilon;
        let f_plus = free_energy_for_kraus(&params_plus.to_matrix(), rho, target);
        grad_real[idx] = (f_plus - base_f) / epsilon;

        let mut params_plus_im = params.clone();
        params_plus_im.imag[idx] += epsilon;
        let f_plus_im = free_energy_for_kraus(&params_plus_im.to_matrix(), rho, target);
        grad_imag[idx] = (f_plus_im - base_f) / epsilon;
    }

    KrausParams { real: grad_real, imag: grad_imag, dim }
}

pub fn optimize_kraus(
    rho: &DMatrix<Complex64>,
    target: &DMatrix<Complex64>,
    dim: usize,
    max_iter: usize,
    learning_rate: f64,
) -> KrausParams {
    let mut params = KrausParams::random(dim);
    for _ in 0..max_iter {
        let grad = numerical_gradient(&params, rho, target, 1e-6);
        params.update(&grad, learning_rate);
    }
    params
}
