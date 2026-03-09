pub struct FacilityNetwork {
    pub aerial: Vec<Span>,
    pub underground: Vec<Span>,
}

impl FacilityNetwork {
    /// Calcular vulnerabilidade
    pub fn vulnerability_index(&self) -> f64 {
        let total = self.aerial.len() + self.underground.len();
        if total == 0 { return 0.0; }

        let aerial_ratio = self.aerial.len() as f64 / total as f64;

        // Aéreo = mais vulnerável
        aerial_ratio
    }
}

pub struct Span {
    pub id: String,
    pub length_m: f64,
}
