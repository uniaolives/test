//! arkhe_crypto::zk_lattice
//! Geração de Provas de Conhecimento Zero Pós-Quânticas via Ring-LWE.
//! Entrelaça a fase física (Termodinâmica) à identidade do Nó (Criptografia).

use rand::rngs::OsRng;
use rand::Rng;

// Parâmetros do Reticulado (Lattice) simplificados para o CubeSat
const Q_MODULUS: i64 = 3329; // Primo para o anel de polinómios (ex: Kyber)
const N_DEGREE: usize = 256; // Grau do polinómio

#[derive(Clone, Debug)]
pub struct Polynomial {
    pub coeffs: [i64; N_DEGREE],
}

impl Polynomial {
    pub fn new() -> Self {
        Polynomial { coeffs: [0; N_DEGREE] }
    }

    /// Adição de polinómios no anel mod Q
    pub fn add(&self, other: &Self) -> Self {
        let mut result = [0; N_DEGREE];
        for i in 0..N_DEGREE {
            result[i] = (self.coeffs[i] + other.coeffs[i]) % Q_MODULUS;
        }
        Polynomial { coeffs: result }
    }

    /// Multiplicação polinomial no anel (simplificada O(n^2) para leitura,
    /// na prática usa-se NTT - Number Theoretic Transform O(n log n))
    pub fn multiply(&self, other: &Self) -> Self {
        let mut result = [0; N_DEGREE];
        for i in 0..N_DEGREE {
            for j in 0..N_DEGREE {
                let index = (i + j) % N_DEGREE;
                // Redução anti-cíclica (X^N + 1 = 0)
                let sign = if i + j >= N_DEGREE { -1 } else { 1 };
                result[index] = (result[index] + sign * self.coeffs[i] * other.coeffs[j]) % Q_MODULUS;
                if result[index] < 0 {
                    result[index] += Q_MODULUS;
                }
            }
        }
        Polynomial { coeffs: result }
    }
}

pub struct ArkheZKProver {
    pub public_matrix_a: Polynomial,
    pub secret_key_s: Polynomial,
}

impl ArkheZKProver {
    pub fn new(public_a: Polynomial, secret_s: Polynomial) -> Self {
        Self {
            public_matrix_a: public_a,
            secret_key_s: secret_s,
        }
    }

    /// Gera o compromisso ZK que entrelaça a fase física medida.
    pub fn generate_phase_proof(&self, thermodynamic_phase: f64) -> Polynomial {
        let mut rng = OsRng;

        // 1. Amostrar o Erro 'e' (Distribuição Gaussiana discreta pequena)
        let mut error_e = [0; N_DEGREE];
        for i in 0..N_DEGREE {
            error_e[i] = rng.gen_range(-2..=2);
        }
        let poly_e = Polynomial { coeffs: error_e };

        // 2. Codificar a fase termodinâmica (0.0 a 2PI) no polinómio
        let mut phase_encoded = [0; N_DEGREE];
        // Injetamos a fase no coeficiente de ordem zero (DC component)
        let scaled_phase = ((thermodynamic_phase / (2.0 * std::f64::consts::PI)) * (Q_MODULUS as f64 / 2.0)) as i64;
        phase_encoded[0] = scaled_phase;
        let poly_phase = Polynomial { coeffs: phase_encoded };

        // 3. A = A * s (Ocultação da chave)
        let hidden_s = self.public_matrix_a.multiply(&self.secret_key_s);

        // 4. Compromisso Final: b = A*s + e + Encode(phi)
        let proof = hidden_s.add(&poly_e).add(&poly_phase);

        proof
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zk_proof_generation() {
        let a = Polynomial { coeffs: [1; N_DEGREE] };
        let s = Polynomial { coeffs: [1; N_DEGREE] };
        let prover = ArkheZKProver::new(a, s);

        let proof = prover.generate_phase_proof(0.618);
        assert_ne!(proof.coeffs[0], 0);
    }
}
