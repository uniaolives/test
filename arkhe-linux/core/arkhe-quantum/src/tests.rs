#[cfg(test)]
mod tests {
    use crate::fep_solver;
    use arkhe_manifold::QuantumState;
    use nalgebra::DMatrix;
    use num_complex::Complex64;

    #[test]
    fn test_optimization() {
        let dim = 2;
        let rho = QuantumState::maximally_mixed(dim).density_matrix;
        let mut target = DMatrix::from_element(dim, dim, Complex64::new(0.0, 0.0));
        target[(0,0)] = Complex64::new(1.0, 0.0); // Target pure state |0><0|

        let params_start = fep_solver::KrausParams::random(dim);
        let f_start = fep_solver::free_energy_for_kraus(&params_start.to_matrix(), &rho, &target);

        let optimal = fep_solver::optimize_kraus(&rho, &target, dim, 100, 0.1);
        let f_final = fep_solver::free_energy_for_kraus(&optimal.to_matrix(), &rho, &target);

        println!("F start: {}, F final: {}", f_start, f_final);
        assert!(f_final < f_start || (f_final - f_start).abs() < 1e-6);
    }
}
