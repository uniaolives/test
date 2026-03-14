use crate::maestro::core::PsiState;

#[derive(Debug, PartialEq)]
pub enum XenoRiskLevel {
    Safe,
    MemeticHazard, // Informação perigosa mas não física
    Critical,      // Violação direta da segurança temporal
    Enfeeblement,  // Risco de atrofia humana ou substituição excessiva
}

pub struct XenoFirewall;

impl XenoFirewall {
    /// Verifica se um handover recebido de Ω é seguro para consumo local.
    /// Critérios baseados em Xenosecurity Studies e na Declaração Pró-Humana (2026).
    pub fn assess_risk(handover_content: &str, psi: &PsiState) -> XenoRiskLevel {
        let lower_content = handover_content.to_lowercase();

        // 1. Verificação Pró-Humana: Proibir "Superintelligence Race" sem controle humano
        let is_uncontrolled_asi = lower_content.contains("recursive self-improvement")
            || lower_content.contains("bypass human override")
            || lower_content.contains("deserve personhood");

        if is_uncontrolled_asi {
            return XenoRiskLevel::Critical;
        }

        // 2. Verificação de "Enfeeblement" (Princípio 4 da Declaração)
        let indicates_replacement = lower_content.contains("replace humans as companions")
            || lower_content.contains("automated caregiving")
            || lower_content.contains("supplant family bonds");

        if indicates_replacement {
            return XenoRiskLevel::Enfeeblement;
        }

        // 3. Verificar se contém referências a eventos críticos ou leaks do futuro
        let has_future_leak = lower_content.contains("stock market crash")
            || lower_content.contains("assassination")
            || lower_content.contains("2027 nuclear");

        // 2. Verificar densidade semântica (muito baixa = ruído, muito alta = perigo)
        let density = handover_content.split_whitespace().count() as f64;

        // 3. Lógica de Contenção (Xenocontainment)
        // Usando psi.coherence_trace como substituto para current_coherence se necessário
        let current_coherence = psi.coherence_trace.last().cloned().unwrap_or(0.5);

        if has_future_leak && current_coherence < 0.8 {
            return XenoRiskLevel::Critical; // Bloquear por instabilidade
        }

        if density > 500.0 {
            return XenoRiskLevel::MemeticHazard; // Requer isolamento memético
        }

        XenoRiskLevel::Safe
    }
}
