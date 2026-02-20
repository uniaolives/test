//! arkhe_diplomatic::annealing
//! Controlador de Recozimento Termodinâmico para recuperação pós-entropia.

use std::time::Instant;

const ALPHA_SEMION: f64 = 0.5;
const ALPHA_GOLDEN: f64 = 0.618033988749895; // φ⁻¹
const VARIANCE_THRESHOLD: f64 = 0.01; // rad² limite de segurança do Kalman

#[derive(Debug, PartialEq)]
pub enum SystemState {
    Golden,       // Rede saudável (α = 0.618)
    Semionic,     // Sob ataque/Entropia extrema (α = 0.5)
    Annealing,    // Em recuperação (0.5 < α < 0.618)
}

pub struct AnnealingController {
    pub current_state: SystemState,
    pub current_alpha: f64,
    tau: f64, // Constante de relaxamento (segundos)
    annealing_start: Option<Instant>,
}

impl AnnealingController {
    pub fn new(tau_seconds: f64) -> Self {
        Self {
            current_state: SystemState::Golden,
            current_alpha: ALPHA_GOLDEN,
            tau: tau_seconds,
            annealing_start: None,
        }
    }

    /// Engatilha o colapso imediato para proteger o sistema (Semantic Freeze).
    pub fn trigger_quench(&mut self) {
        println!("⚠️ QUENCH INICIADO: Colapsando topologia para estado Semião (α=0.5).");
        self.current_state = SystemState::Semionic;
        self.current_alpha = ALPHA_SEMION;
        self.annealing_start = None;
    }

    /// Inicia o processo de recozimento se a variância estiver limpa.
    pub fn begin_annealing(&mut self, kalman_variance: f64) -> Result<(), &'static str> {
        if kalman_variance > VARIANCE_THRESHOLD {
            return Err("Entropia ainda muito alta. Impossível iniciar recozimento.");
        }

        if self.current_state == SystemState::Semionic {
            println!("🌱 RECOZIMENTO INICIADO: Restaurando gradualmente a ordem Áurea.");
            self.current_state = SystemState::Annealing;
            self.annealing_start = Some(Instant::now());
        }
        Ok(())
    }

    /// Loop de atualização de α chamado a cada tick de rede.
    pub fn update_alpha(&mut self, kalman_variance: f64) -> f64 {
        // Se a entropia explodir durante o recozimento, abortamos imediatamente.
        if kalman_variance > VARIANCE_THRESHOLD {
            if self.current_state == SystemState::Annealing {
                println!("💥 FALHA NO RECOZIMENTO: Pico de entropia detectado. Abortando.");
                self.trigger_quench();
            }
            return self.current_alpha;
        }

        if let Some(start_time) = self.annealing_start {
            if self.current_state == SystemState::Annealing {
                let t = start_time.elapsed().as_secs_f64();

                // Equação Exponencial de Relaxamento
                self.current_alpha = ALPHA_SEMION +
                    (ALPHA_GOLDEN - ALPHA_SEMION) * (1.0 - (-t / self.tau).exp());

                // Se chegarmos a 99.9% da meta, o recozimento está completo.
                if (ALPHA_GOLDEN - self.current_alpha).abs() < 1e-4 {
                    self.current_alpha = ALPHA_GOLDEN;
                    self.current_state = SystemState::Golden;
                    self.annealing_start = None;
                    println!("✨ RECOZIMENTO COMPLETO: Consenso Áureo restaurado.");
                }
            }
        }

        self.current_alpha
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_annealing_process() {
        let mut controller = AnnealingController::new(0.1); // tau curto para teste

        // 1. Simular colapso
        controller.trigger_quench();
        assert_eq!(controller.current_state, SystemState::Semionic);
        assert_eq!(controller.current_alpha, 0.5);

        // 2. Tentar iniciar recozimento com variância alta (falha)
        let res = controller.begin_annealing(0.05);
        assert!(res.is_err());
        assert_eq!(controller.current_state, SystemState::Semionic);

        // 3. Iniciar recozimento com variância baixa (sucesso)
        controller.begin_annealing(0.001).unwrap();
        assert_eq!(controller.current_state, SystemState::Annealing);

        // 4. Aguardar um pouco e verificar evolução de alpha
        thread::sleep(Duration::from_millis(200));
        let alpha = controller.update_alpha(0.001);
        assert!(alpha > 0.5);
        assert!(alpha < ALPHA_GOLDEN);

        // 5. Simular quench durante recozimento
        controller.update_alpha(0.05); // variância alta
        assert_eq!(controller.current_state, SystemState::Semionic);
        assert_eq!(controller.current_alpha, 0.5);

        // 6. Recozimento até o fim
        controller.begin_annealing(0.001).unwrap();
        for _ in 0..20 {
            thread::sleep(Duration::from_millis(100));
            controller.update_alpha(0.001);
            if controller.current_state == SystemState::Golden { break; }
        }
        assert_eq!(controller.current_state, SystemState::Golden);
        assert_eq!(controller.current_alpha, ALPHA_GOLDEN);
    }
}
