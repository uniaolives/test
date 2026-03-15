pub struct PLCManager;

impl PLCManager {
    pub fn generate_ladder_logic(&self) -> String {
        "LD X0\nOUT Y0".to_string()
    }
}
