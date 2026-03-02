pub enum HumanAction {
    DamOpen(u32, f64),
    ShipPassage(f64, f64),
}

pub struct HumanIntervention {
    pub action: HumanAction,
}

pub struct IntentionalityDecoder;

impl IntentionalityDecoder {
    pub fn decode_and_apply(&self, _human_data: &HumanIntervention) -> f64 {
        0.78
    }
}
