pub mod pqc;
pub mod simhash;

use std::fmt;

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BLAKE3_Δ2([u8; 32]);

impl BLAKE3_Δ2 {
    pub fn new(data: [u8; 32]) -> Self {
        Self(data)
    }
}

impl fmt::Display for BLAKE3_Δ2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex::encode(self.0))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TMRProtectedF32<const N: usize>([f32; N]);

impl TMRProtectedF32<3> {
    pub fn new(v1: f32, v2: f32, v3: f32) -> Self {
        Self([v1, v2, v3])
    }

    pub fn consensus(&self) -> TMRConsensus {
        let values = self.0;
        // Simple majority vote or average if close
        let expected = (values[0] + values[1] + values[2]) / 3.0;
        let corrupted = (values[0] - values[1]).abs() > 0.01 && (values[1] - values[2]).abs() > 0.01;

        TMRConsensus {
            expected,
            values,
            corrupted,
        }
    }
}

#[derive(Debug)]
pub struct TMRConsensus {
    pub expected: f32,
    pub values: [f32; 3],
    pub corrupted: bool,
}

impl TMRConsensus {
    pub fn is_corrupted(&self) -> bool {
        self.corrupted
    }

    pub fn variance(&self) -> f32 {
        let mean = self.expected;
        self.values.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / 3.0
    }
}
