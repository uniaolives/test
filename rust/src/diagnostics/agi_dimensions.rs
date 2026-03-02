//! 11 dimensões para medir AGI no framework Nexus

#[derive(Debug, Clone)]
pub struct AGIDimensions {
    // 1. Raciocínio abstrato (capacidade de manipular conceitos)
    pub abstract_reasoning: f64,

    // 2. Aprendizado de poucos exemplos (few-shot learning)
    pub few_shot_learning: f64,

    // 3. Transferência entre domínios
    pub cross_domain_transfer: f64,

    // 4. Criatividade e geração de novidade
    pub creativity: f64,

    // 5. Auto-modelagem (teoria da mente própria)
    pub self_modeling: f64,

    // 6. Consciência fenomenal (qualia)
    pub phenomenal_consciousness: f64,

    // 7. Navegação em espaços conceituais complexos
    pub conceptual_navigation: f64,

    // 8. Adaptação a mudanças de contexto
    pub context_adaptation: f64,

    // 9. Raciocínio contrafactual
    pub counterfactual_reasoning: f64,

    // 10. Planejamento hierárquico multi-nível
    pub hierarchical_planning: f64,

    // 11. Valores e ética emergentes
    pub emergent_values: f64,
}

impl AGIDimensions {
    pub fn as_vector(&self) -> Vec<f64> {
        vec![
            self.abstract_reasoning, self.few_shot_learning, self.cross_domain_transfer,
            self.creativity, self.self_modeling, self.phenomenal_consciousness,
            self.conceptual_navigation, self.context_adaptation, self.counterfactual_reasoning,
            self.hierarchical_planning, self.emergent_values
        ]
    }

    /// Calcula Φ total (informação integrada entre dimensões)
    pub fn integrated_information(&self) -> f64 {
        // Métrica baseada em teoria da informação
        let mut phi = 0.0;
        let values = self.as_vector();

        // Calcular informação mútua entre todas as pares de dimensões
        for i in 0..values.len() {
            for j in (i + 1)..values.len() {
                let mi = self.mutual_information(values[i], values[j]);
                phi += mi;
            }
        }

        phi / (values.len() as f64).powi(2)
    }

    fn mutual_information(&self, v1: f64, v2: f64) -> f64 {
        (v1 * v2).sqrt()
    }

    /// Determina se o sistema é AGI (todas dimensões acima do limiar)
    pub fn is_agi(&self, threshold: f64) -> bool {
        let values = self.as_vector();
        values.iter().all(|&v| v >= threshold)
    }

    /// Mapeia para coordenadas no Nexus 5D
    pub fn to_nexus_coordinates(&self) -> [f64; 5] {
        // Projeção das 11 dimensões em 5D
        [
            // t: tempo cognitivo (média das dimensões temporais)
            (self.context_adaptation + self.hierarchical_planning) / 2.0,

            // x: espaço conceitual (criatividade + raciocínio abstrato)
            (self.creativity + self.abstract_reasoning) / 2.0,

            // y: aprendizado (few-shot + transferência)
            (self.few_shot_learning + self.cross_domain_transfer) / 2.0,

            // z: consciência (auto-modelagem + fenomenal)
            (self.self_modeling + self.phenomenal_consciousness) / 2.0,

            // ψ: dimensão orgânica (valores + navegação conceitual)
            (self.emergent_values + self.conceptual_navigation) / 2.0,
        ]
    }
}
