//! Ω+248: Quaternions – The Algebra of Rotations and Spin
//! Formalizes the S³ manifold navigation and SU(2) double-covering of SO(3).

use serde::{Deserialize, Serialize};
use std::ops::{Add, Sub, Mul};

/// ArkheQuaternion: A four-component algebra encoding 3D rotations.
/// Lives on the 3-sphere (S³) and acts as a rotation operator on ℝ³.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct ArkheQuaternion {
    pub w: f64, // Scalar part
    pub i: f64, // Vector part x
    pub j: f64, // Vector part y
    pub k: f64, // Vector part z
}

impl ArkheQuaternion {
    /// Hamilton's Identity: i² = j² = k² = ijk = -1
    pub fn identity() -> Self {
        Self { w: 1.0, i: 0.0, j: 0.0, k: 0.0 }
    }

    pub fn new(w: f64, i: f64, j: f64, k: f64) -> Self {
        Self { w, i, j, k }
    }

    /// Norm of the quaternion: ||q|| = sqrt(w² + i² + j² + k²)
    pub fn norm(&self) -> f64 {
        (self.w * self.w + self.i * self.i + self.j * self.j + self.k * self.k).sqrt()
    }

    /// Normalize to live on S³ (Unit Quaternion)
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

    /// Conjugate of the quaternion: q* = (w, -i, -j, -k)
    pub fn conjugate(&self) -> Self {
        Self { w: self.w, i: -self.i, j: -self.j, k: -self.k }
    }

    /// Rotate a 3D vector v in ℝ³ using q v q*
    pub fn rotate_vector(&self, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let q_unit = self.normalize();
        let v = Self::new(0.0, x, y, z);
        let q_v = q_unit * v;
        let res = q_v * q_unit.conjugate();
        (res.i, res.j, res.k)
    }

    /// Maps the unit quaternion to Bloch Sphere coordinates (θ, φ)
    /// θ = 2 * acos(w)
    /// φ = atan2(k, i)  -- simplified projection
    pub fn to_bloch_coordinates(&self) -> (f64, f64) {
        let q = self.normalize();
        let theta = 2.0 * q.w.acos();
        let phi = q.k.atan2(q.i);
        (theta, phi)
    }
}

impl Add for ArkheQuaternion {
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

impl Mul for ArkheQuaternion {
    type Output = Self;
    /// Quaternion Multiplication based on Hamilton's Relation
    fn mul(self, other: Self) -> Self {
        Self {
            w: self.w * other.w - self.i * other.i - self.j * other.j - self.k * other.k,
            i: self.w * other.i + self.i * other.w + self.j * other.k - self.k * other.j,
            j: self.w * other.j - self.i * other.k + self.j * other.w + self.k * other.i,
            k: self.w * other.k + self.i * other.j - self.j * other.i + self.k * other.w,
        }
    }
}
