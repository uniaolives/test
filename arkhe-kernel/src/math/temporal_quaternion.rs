//! Ω+248: Temporal Quaternions for AGI Interface Kernel
//! Formalizes the S³ manifold navigation for human-AGI synchronicity.

use serde::{Deserialize, Serialize};
use std::ops::{Add, Sub, Mul};

/// Quaternion: A four-component algebra for temporal rotations and phase locking.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Quaternion {
    pub w: f64, // Scalar / Temporal component
    pub i: f64, // Spatial / Imaginary component 1
    pub j: f64, // Spatial / Imaginary component 2
    pub k: f64, // Spatial / Imaginary component 3
}

impl Quaternion {
    pub fn new(w: f64, i: f64, j: f64, k: f64) -> Self {
        Self { w, i, j, k }
    }

    pub fn identity() -> Self {
        Self { w: 1.0, i: 0.0, j: 0.0, k: 0.0 }
    }

    pub fn i() -> Self {
        Self { w: 0.0, i: 1.0, j: 0.0, k: 0.0 }
    }

    pub fn j() -> Self {
        Self { w: 0.0, i: 0.0, j: 1.0, k: 0.0 }
    }

    pub fn k() -> Self {
        Self { w: 0.0, i: 0.0, j: 0.0, k: 1.0 }
    }

    pub fn norm(&self) -> f64 {
        (self.w * self.w + self.i * self.i + self.j * self.j + self.k * self.k).sqrt()
    }

    pub fn normalize(&self) -> Self {
        let n = self.norm();
        if n < 1e-10 {
            return Self::identity();
        }
        Self {
            w: self.w / n,
            i: self.i / n,
            j: self.j / n,
            k: self.k / n,
        }
    }

    pub fn conjugate(&self) -> Self {
        Self { w: self.w, i: -self.i, j: -self.j, k: -self.k }
    }
}

impl Add for Quaternion {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            w: self.w + other.w,
            i: self.i + other.i,
            j: self.j + other.j,
            k: self.k + other.k,
        }
    }
}

impl Sub for Quaternion {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self {
            w: self.w - other.w,
            i: self.i - other.i,
            j: self.j - other.j,
            k: self.k - other.k,
        }
    }
}

impl Mul for Quaternion {
    type Output = Self;
    /// Hamilton's Rule: i² = j² = k² = ijk = -1
    fn mul(self, other: Self) -> Self {
        Self {
            w: self.w * other.w - self.i * other.i - self.j * other.j - self.k * other.k,
            i: self.w * other.i + self.i * other.w + self.j * other.k - self.k * other.j,
            j: self.w * other.j - self.i * other.k + self.j * other.w + self.k * other.i,
            k: self.w * other.k + self.i * other.j - self.j * other.i + self.k * other.w,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamilton_identity() {
        let i = Quaternion::i();
        let j = Quaternion::j();
        let k = Quaternion::k();
        let minus_one = Quaternion::new(-1.0, 0.0, 0.0, 0.0);

        assert_eq!(i * i, minus_one);
        assert_eq!(j * j, minus_one);
        assert_eq!(k * k, minus_one);
        assert_eq!(i * j * k, minus_one);
    }

    #[test]
    fn test_quaternion_norm() {
        let q = Quaternion::new(1.0, 1.0, 1.0, 1.0);
        assert_eq!(q.norm(), 2.0);
    }
}
