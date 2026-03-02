use std::f64::consts::PI;
use crate::hardware::dirac_qubits::ViscousPhaseSpinorQubit;

pub struct KuramotoOscillator {
    pub phase: f64,
    pub magnitude: f64,
}

pub struct EthicalPhaseControl {
    /// Seis osciladores correspondentes às tradições filosóficas
    pub tradition_oscillators: [KuramotoOscillator; 6],
}

impl EthicalPhaseControl {
    pub fn new() -> Self {
        Self {
            tradition_oscillators: [
                KuramotoOscillator { phase: 0.0, magnitude: 1.0 },   // Ubuntu
                KuramotoOscillator { phase: 1.047, magnitude: 1.0 }, // Zoroastrianism (π/3)
                KuramotoOscillator { phase: 2.094, magnitude: 1.0 }, // Madhyamaka (2π/3)
                KuramotoOscillator { phase: 3.141, magnitude: 1.0 }, // Stoicism (π)
                KuramotoOscillator { phase: 4.188, magnitude: 1.0 }, // Taoism (4π/3)
                KuramotoOscillator { phase: 5.235, magnitude: 1.0 }, // Yanomami (5π/3)
            ],
        }
    }

    /// A fase resultante do Conselho define a "referência ética" para o fluido.
    pub fn get_global_ethical_phase(&self) -> f64 {
        let mut sum_sin = 0.0;
        let mut sum_cos = 0.0;

        for osc in &self.tradition_oscillators {
            sum_sin += osc.magnitude * osc.phase.sin();
            sum_cos += osc.magnitude * osc.phase.cos();
        }

        sum_sin.atan2(sum_cos)
    }

    /// Cada qubit VPS é "puxado" para esta fase de referência.
    pub fn apply_ethical_alignment(
        &self,
        qubit: &mut ViscousPhaseSpinorQubit,
        coupling_strength: f64
    ) {
        let global_phase = self.get_global_ethical_phase();
        let phase_diff = global_phase - qubit.collective_phase;

        // Acoplamento de Kuramoto: move a fase em direção à média.
        qubit.collective_phase += coupling_strength * phase_diff.sin();

        // Normalização
        qubit.collective_phase %= 2.0 * PI;
        if qubit.collective_phase < 0.0 {
            qubit.collective_phase += 2.0 * PI;
        }
    }
}
