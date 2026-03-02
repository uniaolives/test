#[derive(Debug, Clone, PartialEq)]
pub enum Trait {
    HighEpistemicTrust,
    CeticismoSaudavel,
    ConfiancaRapida,
}

#[derive(Debug, Clone)]
pub struct InheritableTrait {
    pub trait_type: Trait,
    pub expression_level: f32,
}

impl InheritableTrait {
    pub fn mutate_slightly(&self, rate: f32) -> Self {
        // Mock mutation logic
        Self {
            trait_type: self.trait_type.clone(),
            expression_level: self.expression_level * (1.0 + rate),
        }
    }
}
