// arkhe-axos-instaweb/src/cy_utils.rs
use pyo3::prelude::*;
use nalgebra::DMatrix;

#[pyfunction]
pub fn compute_coherence(metric: Vec<Vec<f64>>) -> PyResult<f64> {
    let rows = metric.len();
    if rows == 0 {
        return Ok(0.0);
    }
    let cols = metric[0].len();
    if rows != cols {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Matrix must be square"));
    }

    // Note: DMatrix::from_vec expects column-major by default if using from_vec
    // But we can use from_row_slice or just handle it.
    let flat_metric: Vec<f64> = metric.into_iter().flatten().collect();
    let matrix = DMatrix::from_row_slice(rows, cols, &flat_metric);

    let eigenvals = matrix.complex_eigenvalues();

    let sum_norms: f64 = eigenvals.iter().map(|c| (c.re * c.re + c.im * c.im).sqrt()).sum();
    let coherence = sum_norms / rows as f64;

    Ok(coherence)
}

#[pymodule]
pub fn cy_utils(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_coherence, m)?)?;
    Ok(())
}
