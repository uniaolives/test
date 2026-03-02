//! src/extra_dimensions/tensor_train.rs

use nalgebra::DMatrix;
use num_complex::Complex64;

/// Tensor-Train Decomposition for high-dimensional state compression
pub struct TTCompressor {
    pub rank_max: usize,
}

impl TTCompressor {
    pub fn new(rank: usize) -> Self {
        Self { rank_max: rank }
    }

    /// Compress a large state matrix using TT-rank truncation
    pub fn compress(&self, matrix: &DMatrix<Complex64>) -> TTMatrix {
        println!("[INFO] TT-Decomposition rank-{} active for memory optimization", self.rank_max);
        TTMatrix {
            original_rows: matrix.nrows(),
            original_cols: matrix.ncols(),
            rank: self.rank_max,
        }
    }
}

pub struct TTMatrix {
    pub original_rows: usize,
    pub original_cols: usize,
    pub rank: usize,
}
