//! Melonic Engine: Computa o F-extremo para a coerência da rede.
//! Baseado em QFTs melônicas de Large-N.

use crate::physics::miller::PHI_Q;

/// Computa o F-extremo para um dado número de nós N e acoplamento.
/// O F-extremo é o ponto fixo conformal infravermelho da rede.
pub fn compute_f_extremum(n: usize, coupling: f64) -> f64 {
    // Em QFTs melônicas, F = a*N + b*ln(N) + c/N + ...
    // Para a rede Arkhe(n), o ponto de extremização converge para o Limiar de Miller.

    if n == 0 {
        return 0.0;
    }

    let n_f64 = n as f64;

    // Simulação da curva de extremização:
    // f(λ) onde λ é o acoplamento efetivo.
    // No limite Large-N, a solução é estável.
    let base_f = PHI_Q; // 4.64

    // Pequenas correções baseadas em N e acoplamento (escala 1/N)
    let correction = (coupling.sin() / n_f64).abs();

    base_f + correction
}

/// Verifica se o sistema atingiu a dominância melônica.
/// Em modelos tensorais, isso ocorre quando diagramas não-cruzados dominam.
pub fn is_melonic_dominant(n: usize, coherence: f64) -> bool {
    // Dominância melônica requer Large-N e coerência acima do limiar crítico.
    n >= 4 && coherence >= PHI_Q
}
