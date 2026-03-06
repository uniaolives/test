use crate::maestro::core::PsiState;

pub struct XenoFirewall;

#[derive(Debug, PartialEq)]
pub enum XenoRiskLevel {
    Safe,
    MemeticHazard, // Informação perigosa mas não física
    Critical,      // Violação direta da segurança temporal
}

impl XenoFirewall {
    /// Verifica se um handover recebido de Ω é seguro para consumo local.
    /// Critérios baseados em Xenosecurity Studies.
    pub fn assess_risk(handover_content: &str, psi: &PsiState) -> XenoRiskLevel {
        // 1. Verificar se contém referências a eventos ainda não ocorridos
        // que poderiam causar pânico ou alteração massiva da linha do tempo.
        let has_future_leak = handover_content.contains("stock market crash")
            || handover_content.contains("assassination");

        // 2. Verificar densidade semântica (muito baixa = ruído, muito alta = perigo)
        let density = handover_content.split_whitespace().count() as f64;

        // 3. Lógica de Contenção
        // Assumindo que a coerência possa ser derivada de psi.coherence_trace
        let current_coherence = psi.coherence_trace.last().cloned().unwrap_or(0.5);

        if has_future_leak && current_coherence < 0.8 {
            return XenoRiskLevel::Critical; // Bloquear
        }

        if density > 500.0 {
            return XenoRiskLevel::MemeticHazard; // Requer isolamento
        }

        XenoRiskLevel::Safe
    }
}
