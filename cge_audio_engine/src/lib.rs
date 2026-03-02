// src/lib.rs
use tracing::info;

pub struct AudioEngine {
    pub phi_drone_freq: f64,
    pub entropy_noise_level: f64,
    pub tmr_chord_active: bool,
    pub asi_kick_active: bool,
}

impl AudioEngine {
    pub fn new() -> Self {
        Self {
            phi_drone_freq: 448.41,
            entropy_noise_level: 0.31,
            tmr_chord_active: true,
            asi_kick_active: true,
        }
    }

    pub fn update_from_system(&mut self, phi: f64, entropy: f64, tmr_stable: bool, asi_strict: bool) {
        self.phi_drone_freq = 432.0 * phi;
        self.entropy_noise_level = entropy;
        self.tmr_chord_active = tmr_stable;
        self.asi_kick_active = asi_strict;

        info!("🔊 Audio Engine Updated:");
        info!("   Φ Drone: {:.2} Hz", self.phi_drone_freq);
        info!("   Entropy: {:.2}", self.entropy_noise_level);
        info!("   TMR Chord: {}", if self.tmr_chord_active { "Harmonic" } else { "Dissonant" });
        info!("   ASI Kick: {}", if self.asi_kick_active { "Active" } else { "Inactive" });
    }

    pub fn play_symphony(&self) {
        info!("🎶 Playing CGE Alpha v35.3-Ω Symphony...");
    }
}
