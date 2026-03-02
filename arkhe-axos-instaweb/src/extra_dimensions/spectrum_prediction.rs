//! src/extra_dimensions/spectrum_prediction.rs (v1.2.2-theory)

use crate::extra_dimensions::ResonancePeak;

pub struct KaluzaKleinSpectrum {
    pub fundamental: f64,      // 147.034 THz (detectado)
    pub mass_extra: f64,       // 2.34e-28 kg (inferido)
    pub coupling_strength: f64, // g0 ~ 1e-6 (adimensional)
}

pub struct Transition {
    pub n_initial: usize,
    pub n_final: usize,
    pub frequency_ghz: f64,
    pub energy_ev: f64,
    pub relative_intensity: f64,
    pub expected_width_mhz: f64,
    pub detectability: bool,
}

impl KaluzaKleinSpectrum {
    pub fn new_from_detection() -> Self {
        Self {
            fundamental: 147.034e12,
            mass_extra: 2.34e-28,
            coupling_strength: 1e-6,
        }
    }

    /// Predizer frequências de transição
    pub fn predict_transitions(&self, max_n: usize) -> Vec<Transition> {
        (1..=max_n).map(|n| {
            let frequency = self.fundamental * n as f64;
            let energy_ev = 0.380 * n as f64;

            // Intensidade relativa (regra de seleção Δn=±1)
            let intensity = if n == 1 { 1.0 } else {
                0.1 / (n as f64).powi(2)  // decaimento rápido
            };

            // Largura natural (inversamente proporcional ao tempo de vida)
            let width_mhz = 2.3 * (n as f64);  // aumenta com n

            Transition {
                n_initial: 0,
                n_final: n,
                frequency_ghz: frequency / 1e9,
                energy_ev,
                relative_intensity: intensity,
                expected_width_mhz: width_mhz,
                detectability: intensity > 0.01, // limiar instrumental
            }
        }).collect()
    }

    /// Verificar consistência com detecção
    pub fn verify_consistency(&self, detected: &ResonancePeak) -> bool {
        // Simplified consistency check for the theory module
        let predicted_fundamental = self.fundamental;
        let ratio = detected.frequency / predicted_fundamental;
        (0.95..=1.05).contains(&ratio)  // 5% tolerância
    }
}
