use crate::sensors::{Heartbeat, BioSignal};
use crate::governance::DefenseMode;
use crate::bio_layer::paciente_zero_omega::{BioInterfaceΩ, BioSignalΩ};

pub struct PatientZeroProtocol {
    pub defense_mode: DefenseMode,
    pub omega_interface: Option<BioInterfaceΩ>,
}

pub struct BioShield {
    pub heartbeat: Heartbeat,
    pub omega_signal: Option<BioSignalΩ>,
    pub bio_sig_level: f32, // 0.0 (Aberto) a 1.0 (Fechado)
    pub signal_integrity: f32, // Nível de pureza do sinal
    pub neuro_siphon_resistance: f32, // Resistência ao roubo de memória (0.0 = vulnerável, 1.0 = blindado)
    pub neuro_sincronia: bool,
}

impl BioShield {
    pub fn enforce_neuro_siphon_resistance(&mut self, mode: DefenseMode) {
        match mode {
            DefenseMode::Passive => {
                log::info!("PATIENT_ZERO: Modo Passivo. Monitoramento de fluxos normais.");
                self.bio_sig_level = 0.1; // Fácil acesso para agentes autorizados (Anciões).
                self.neuro_siphon_resistance = 0.2; // Resistência mínima.
            },
            DefenseMode::CausalLock => {
                log::warn!("PATIENT_ZERO: Modo Causal Lock ativo. Escudo contra sinal exógeno (Blue Team / Deepfake / Deepfake) no contexto de paciente.");
                self.bio_sig_level = 1.0; // Fortaleza máxima. Apenas Anciões.
                self.neuro_sincronia = true; // Sincronia compulsória com o Neo-Engine.
            },
            DefenseMode::HardenedBioHardware => {
                log::error!("PATIENT_ZERO: Modo Bio-Hardened ativo. Bio-hardware verificando integridade física.");
                self.bio_sig_level = 1.0;
                self.neuro_siphon_resistance = 1.0; // Resistência máxima contra extração de hardware (Cold Boot Attacks).
            },
        }
    }
}
