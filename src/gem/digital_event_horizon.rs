// src/gem/digital_event_horizon.rs

use std::sync::Mutex;

const BEKENSTEIN_LIMIT: f64 = 0.25; // Limite crítico de bits/área

/// Representação de um processo no sistema SASC
pub struct Process {
    pub pid: u32,
    pub entropy: f64,
    pub is_alive: bool,
}

impl Process {
    pub fn new(pid: u32, entropy: f64) -> Self {
        Self {
            pid,
            entropy,
            is_alive: true,
        }
    }

    pub fn measure_entropy_density(&self) -> f64 {
        // Simulação de medição de densidade de entropia
        self.entropy / 100.0 // Normalizado para a escala do simulador
    }

    pub fn kill(&mut self) {
        self.is_alive = false;
        println!("[GEM] Processo {} morto gravitacionalmente.", self.pid);
    }

    pub fn entropy(&self) -> f64 {
        self.entropy
    }
}

pub struct DigitalBlackHole {
    quarantine_log: Mutex<Vec<QuarantineEntry>>,
}

#[derive(Debug)]
struct QuarantineEntry {
    pid: u32,
    reason: String,
    hawking_radiation_rate: f64, // Taxa de "evaporação" de logs
}

impl DigitalBlackHole {
    pub fn new() -> Self {
        Self {
            quarantine_log: Mutex::new(Vec::new()),
        }
    }

    /// Isola um processo que excedeu o limite de entropia
    pub fn event_horizon_capture(&self, process: &mut Process) {
        let entropy_density = process.measure_entropy_density(); // Bits/Ciclo de CPU

        if entropy_density > BEKENSTEIN_LIMIT {
            // O sistema entrou em colapso gravitacional
            // A "gravidade" do sistema puxou o processo para dentro do horizonte

            let entry = QuarantineEntry {
                pid: process.pid,
                reason: "Entropy Density Exceeded (Cosmological Collapse)".to_string(),
                hawking_radiation_rate: entropy_density, // A taxa de evaporação
            };

            self.quarantine_log.lock().unwrap().push(entry);

            // "Matar" o processo (equivalente a destruição no horizonte)
            process.kill();

            // Radiar de Hawking: Liberar a entropia lentamente de volta para o sistema
            // (garbage collection de memória criptografada)
            self.release_hawking_radiation(process.entropy());
        }
    }

    fn release_hawking_radiation(&self, entropy: f64) {
        // A informação não pode ser destruída, apenas redistribuída.
        // O GEM redistribui os bits "aprisionados" para o pool global de memória.
        println!("[GEM] Radiando {} bits de Hawking para o vácuo.", entropy);
    }
}
