use nalgebra::DMatrix;
use num_complex::Complex64;

/// Um canal quântico definido por um conjunto de operadores de Kraus {K_k}.
/// A ação sobre um estado ρ é: ρ' = Σ_k K_k ρ K_k†
#[derive(Debug, Clone)]
pub struct KrausChannel {
    pub operators: Vec<DMatrix<Complex64>>,
}

impl KrausChannel {
    /// Cria um canal unitário (apenas um operador, a matriz U).
    pub fn unitary(u: DMatrix<Complex64>) -> Self {
        KrausChannel { operators: vec![u] }
    }

    /// Cria um canal de despolarização: ρ' = (1-p)ρ + p I/dim.
    pub fn depolarizing(dim: usize, p: f64) -> Self {
        let identity = DMatrix::from_diagonal_element(dim, dim, Complex64::new(1.0, 0.0));
        let k0 = identity.clone() * Complex64::new((1.0 - p).sqrt(), 0.0);
        let k1 = identity * Complex64::new((p / (dim as f64)).sqrt(), 0.0);
        KrausChannel { operators: vec![k0, k1] }
    }

    /// Aplica o canal a um estado, retornando nova matriz densidade.
    pub fn apply(&self, rho: &DMatrix<Complex64>) -> DMatrix<Complex64> {
        let dim = rho.nrows();
        let mut new_rho = DMatrix::from_element(dim, dim, Complex64::new(0.0, 0.0));
        for k in &self.operators {
            new_rho += k * rho * k.adjoint();
        }
        new_rho
    }

    /// Verifica se o canal é completamente positivo e preserva traço (CPTP).
    pub fn is_cptp(&self) -> bool {
        let dim = self.operators[0].nrows();
        let mut sum = DMatrix::from_element(dim, dim, Complex64::new(0.0, 0.0));
        for k in &self.operators {
            sum += k.adjoint() * k;
        }
        let identity = DMatrix::identity(dim, dim);
        (sum - identity).norm() < 1e-10
    }
}
