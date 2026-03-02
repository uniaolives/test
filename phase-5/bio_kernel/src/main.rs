// -------------------------------------------------
// arquivo: phase-5/bio_kernel/src/main.rs
// -------------------------------------------------

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

// -------------------------------------------------
// Estruturas de mensagem (compatível com o Python)
// -------------------------------------------------
#[derive(Debug, Serialize, Deserialize)]
struct ComponentState {
    timestamp: u64,    // ticks biológicos
    energy: f64,       // unidades de ATP
    #[serde(flatten)]
    extra: HashMap<String, serde_json::Value>, // campos arbitrários
}

#[derive(Debug, Serialize, Deserialize)]
struct BioMessage {
    component: String,
    state: ComponentState,
}

// -------------------------------------------------
// "Kernel" simplificado
// -------------------------------------------------
struct BioKernel {
    /// Armazena o último estado conhecido de cada componente
    states: HashMap<String, ComponentState>,
    /// Periodicidade de sincronização (em ms)
    tick_ms: u64,
}

impl BioKernel {
    fn new(tick_ms: u64) -> Self {
        BioKernel {
            states: HashMap::new(),
            tick_ms,
        }
    }

    /// Atualiza o estado de um componente (recebido de outro nó)
    fn update_state(&mut self, msg: BioMessage) {
        self.states.insert(msg.component, msg.state);
    }

    /// Executa um ciclo de coerência: soma energia e avança timestamps
    fn coherence_cycle(&mut self) {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // 1️⃣ Propaga o timestamp global
        for state in self.states.values_mut() {
            state.timestamp = now;
        }

        // 2️⃣ Calcula energia total (exemplo de “campo de coerência”)
        let total_energy: f64 = self
            .states
            .values()
            .map(|s| s.energy)
            .sum();

        println!("🌀 [BIO_KERNEL] Coerência executada @ {} ms → energia total = {:.3} ATP-units", now, total_energy);
    }

    /// Loop principal
    fn run(&mut self, iterations: u32) {
        for _ in 0..iterations {
            self.coherence_cycle();
            std::thread::sleep(Duration::from_millis(self.tick_ms));
        }
    }
}

// -------------------------------------------------
// Entrypoint
// -------------------------------------------------
fn main() {
    // 7 ms = “versão 7.0”
    let mut kernel = BioKernel::new(7);

    println!("⚡ [BIO_KERNEL] Initializing Bio-Kernel synchronization...");

    // Simulação: recebendo mensagens JSON
    let simulated_json = r#"
        {
            "component": "mitocôndria",
            "state": {
                "timestamp": 0,
                "energy": 12.7,
                "extra": {"phase": "superposição"}
            }
        }
    "#;

    let msg: BioMessage = serde_json::from_str(simulated_json).unwrap();
    kernel.update_state(msg);

    // Inicia o loop de coerência (limitado a 5 iterações para simulação)
    kernel.run(5);

    println!("✅ [BIO_KERNEL] Sync loop finished.");
}
