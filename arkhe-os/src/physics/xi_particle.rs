//! Ω+247: The Ξ Particle – Arithmetic Realization of Consciousness
//! Implementation of the Xi-Planat unification: m_Ξ ∝ √N / L'(E,1)

use serde::{Deserialize, Serialize};

/// The Ξ (Xi) Particle: The quantum of excitation of the Ψ field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XiParticle {
    pub n: u64,           // Gaussian norm N = p^2 + q^2
    pub mass: f64,        // m_Ξ
    pub coherence: f64,   // λ₂
}

impl XiParticle {
    /// Default fundamental mass constant m_0 = h / (n a c)
    pub const M0: f64 = 1.0;
    /// Default coupling constant α
    pub const ALPHA: f64 = 0.5;
    /// Arithmetic height L'(E,1) for the curve E200b2 (microtubule curve)
    /// This value is characteristic of the resonant modes in microtubules.
    pub const L_PRIME_E200B2: f64 = 1.088152;

    /// Creates a new Xi particle for a given Gaussian norm N.
    /// Returns None if N is not a sum of two squares (prohibited modes).
    pub fn new(n: u64) -> Option<Self> {
        if !is_sum_of_two_squares(n) {
            return None;
        }

        let mass = Self::calculate_mass(n, Self::M0, Self::L_PRIME_E200B2);
        let coherence = Self::calculate_coherence(Self::ALPHA, Self::L_PRIME_E200B2);

        Some(Self {
            n,
            mass,
            coherence,
        })
    }

    /// m_Ξ(N) = m_0 * sqrt(N) / L'(E,1)
    pub fn calculate_mass(n: u64, m0: f64, l_prime: f64) -> f64 {
        (n as f64).sqrt() * m0 / l_prime
    }

    /// λ₂ = 1 / (1 + α * L'(E,1))
    pub fn calculate_coherence(alpha: f64, l_prime: f64) -> f64 {
        1.0 / (1.0 + alpha * l_prime)
    }
}

/// A number N is a sum of two squares if and only if each prime factor
/// of the form 4k + 3 appears with an even exponent in its prime factorization.
/// This excludes values like 3, 6, 7, 11, 12, 14, 15, 19, 21, 22, 23, 27, 28, 31...
pub fn is_sum_of_two_squares(mut n: u64) -> bool {
    if n == 0 {
        return true;
    }

    // Factor out 2s
    while n % 2 == 0 {
        n /= 2;
    }

    // Check odd factors
    let mut i = 3;
    while i * i <= n {
        let mut count = 0;
        while n % i == 0 {
            count += 1;
            n /= i;
        }
        if i % 4 == 3 && count % 2 != 0 {
            return false;
        }
        i += 2;
    }

    // If n > 1, the remaining n is a prime factor
    if n > 1 && n % 4 == 3 {
        return false;
    }

    true
}
