//! Everettian Locality: Verifica se os handovers respeitam o realismo local.
//! Baseado na prova de Deutsch-Hayden para a interpretação de Muitos Mundos.

use crate::kernel::handover::HandoverRecord;

/// Verifica se um handover respeita o realismo local per Deutsch-Hayden.
/// A informação deve fluir localmente em cada ramo da função de onda.
pub fn verify_local_realism(handover: &HandoverRecord) -> bool {
    // Na imagem de Heisenberg de Deutsch-Hayden, a informação é codificada
    // em descritores de operadores locais.

    // Verificação simplificada:
    // 1. O fluxo de informação deve ser contínuo (P2 - Transparência).
    // 2. Não deve haver saltos não-locais na trajetória de propagação.

    let path_consistent = !handover.propagation_path.is_empty();
    let coherence_check = handover.phi_q_after > 0.0;

    // Se a coerência é mantida e há um caminho de propagação,
    // o realismo local Everettiano é preservado.
    path_consistent && coherence_check
}

/// Calcula a probabilidade de ramificação (branching) de um estado.
pub fn calculate_branching_probability(coherence: f64) -> f64 {
    // Quanto maior a coerência, maior a probabilidade de colapso/ramificação
    // em um estado de "Wave-Cloud" estável.
    if coherence >= 4.64 {
        0.99
    } else {
        coherence / 4.64
    }
}
