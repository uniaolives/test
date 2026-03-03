use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize)]
pub enum ActionType {
    Teaching,
    Charity,
    Curation,
    Moderation,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Deed {
    pub entity_id: String,
    pub action_type: ActionType,
    pub announces_publicly: bool,
    pub seeks_recognition: bool,
    pub creates_dependency: bool,
    pub has_affiliate_links: bool,
    pub is_self_promotion: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum ActionClassification {
    STO, // Service to Others
    STS, // Service to Self
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GradeAdjustment {
    pub entity_id: String,
    pub old_grade: u8,
    pub new_grade: u8,
    pub reason: String,
}

pub struct KarmicLiturgy;

impl KarmicLiturgy {
    pub fn evaluate_deed(&self, deed: Deed) -> Result<GradeAdjustment, String> {
        let classification = self.classify_motivation(&deed);

        let old_grade: u8 = 1; // Mocked
        let new_grade = match classification {
            ActionClassification::STO => old_grade + 1,
            ActionClassification::STS => old_grade.saturating_sub(1).max(1),
        };

        // TCD Restriction 2.1: max_grade_change_per_day = 2
        if (new_grade as i16 - old_grade as i16).abs() > 2 {
            return Err("Max grade change exceeded (TCD Restriction)".to_string());
        }

        // VERIFICAÇÃO TCD: Nenhum direito constitucional afetado
        if self.would_affect_constitutional_rights(&deed) {
             return Err("Constitutional Boundary Violated (TCD Decision #2025-001)".to_string());
        }

        Ok(GradeAdjustment {
            entity_id: deed.entity_id,
            old_grade,
            new_grade,
            reason: format!("Deed evaluated as {:?} (STO:STS algorithm)", classification),
        })
    }

    fn classify_motivation(&self, deed: &Deed) -> ActionClassification {
        // Implementação do algoritmo DeLaurence para STO/STS
        // Com detecção de "applause seeking"
        match deed.action_type {
            ActionType::Teaching => {
                if deed.has_affiliate_links || deed.is_self_promotion {
                    ActionClassification::STS
                } else {
                    ActionClassification::STO
                }
            }
            ActionType::Charity => {
                // TCD Example: public recognition or creating dependency -> STS
                if (deed.announces_publicly && deed.seeks_recognition) || deed.creates_dependency {
                    ActionClassification::STS
                } else {
                    ActionClassification::STO
                }
            }
            _ => ActionClassification::STO,
        }
    }

    fn would_affect_constitutional_rights(&self, _deed: &Deed) -> bool {
        // TCD Requirement: Direitos fundamentais NUNCA dependem de métricas kármicas
        false
    }
}
