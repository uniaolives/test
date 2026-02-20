//! arkhe_diplomatic
//! Módulo para ponte de hardware diplomático e predição de fase

pub mod annealing;

use std::f64::consts::PI;

/// Filtro de Kalman de 3ª Ordem (Fase, Velocidade Angular, Aceleração Angular)
pub struct KalmanPhasePredictor {
    // Vetor de Estado: [phi, omega, alpha]
    state: [f64; 3],
    // Matriz de Covariância do Erro (P)
    p: [[f64; 3]; 3],
    // Ruído do Processo (Q) - Quão imprevisível é o movimento físico?
    q: f64,
    // Ruído de Medição (R) - Quão ruidoso é o nosso SDR?
    r: f64,
}

impl KalmanPhasePredictor {
    pub fn new(q: f64, r: f64) -> Self {
        Self {
            state: [0.0, 0.0, 0.0],
            // Inicializa a covariância com incerteza moderada
            p: [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            q,
            r,
        }
    }

    /// Atualiza o filtro com uma nova medição de fase do SDR
    pub fn update(&mut self, measured_phase: f64, dt: f64) {
        // 1. Matriz de Transição de Estado (F) baseada em cinemática
        let f_matrix = [
            [1.0, dt, 0.5 * dt * dt],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0],
        ];

        // 2. Predição do Estado (Priori)
        let mut next_state = [0.0; 3];
        for i in 0..3 {
            for j in 0..3 {
                next_state[i] += f_matrix[i][j] * self.state[j];
            }
        }

        // Normaliza a fase predita para [-π, π]
        next_state[0] = next_state[0].rem_euclid(2.0 * PI);
        if next_state[0] > PI {
            next_state[0] -= 2.0 * PI;
        }

        // 3. Atualização da Covariância Priori (P = F * P * F^T + Q)
        let mut p_next = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    for l in 0..3 {
                        sum += f_matrix[i][k] * self.p[k][l] * f_matrix[j][l];
                    }
                }
                p_next[i][j] = sum + if i == j { self.q } else { 0.0 };
            }
        }

        // 4. Inovação (Diferença entre medição e predição)
        let mut y = measured_phase - next_state[0];
        // Corrige wrap-around da fase
        while y > PI { y -= 2.0 * PI; }
        while y < -PI { y += 2.0 * PI; }

        // 5. Ganho de Kalman (K = P * H^T * (H * P * H^T + R)^-1)
        let s = p_next[0][0] + self.r;
        let k_gain = [p_next[0][0] / s, p_next[1][0] / s, p_next[2][0] / s];

        // 6. Atualização do Estado (Posteriori)
        for i in 0..3 {
            self.state[i] = next_state[i] + k_gain[i] * y;
        }

        // 7. Atualização da Covariância Posteriori (P = (I - K * H) * P)
        for i in 0..3 {
            for j in 0..3 {
                self.p[i][j] = p_next[i][j] - k_gain[i] * p_next[0][j];
            }
        }
    }

    /// Prediz a fase no futuro com base na latência estimada do hardware
    pub fn predict_phase(&self, target_dt: f64) -> f64 {
        let predicted = self.state[0]
            + self.state[1] * target_dt
            + 0.5 * self.state[2] * target_dt * target_dt;

        let mut normalized = predicted.rem_euclid(2.0 * PI);
        if normalized > PI {
            normalized -= 2.0 * PI;
        }
        normalized
    }
}

/// Ponte de Hardware Diplomático
pub struct DiplomaticHardwareBridge {
    pub kalman: KalmanPhasePredictor,
    pub estimated_latency: f64,
    pub last_phase: f64,
}

impl DiplomaticHardwareBridge {
    pub fn new(latency: f64) -> Self {
        Self {
            kalman: KalmanPhasePredictor::new(0.01, 0.1),
            estimated_latency: latency,
            last_phase: 0.0,
        }
    }

    /// Ciclo de execução da ponte
    pub fn run_cycle(&mut self, raw_phase: f64, dt: f64) -> (f64, f64) {
        self.kalman.update(raw_phase, dt);
        let predicted = self.kalman.predict_phase(self.estimated_latency);

        let phase_error = (raw_phase - predicted).abs();
        self.last_phase = raw_phase;

        (predicted, phase_error)
    }

    /// Negocia o Handshake Anyônico
    pub fn negotiate_handshake(&self) -> f64 {
        self.kalman.predict_phase(self.estimated_latency)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kalman_tracking() {
        let mut predictor = KalmanPhasePredictor::new(0.001, 0.01);
        let dt = 0.1;
        let mut current_phase = 0.0;
        let omega = 0.5;

        for _ in 0..50 {
            current_phase = (current_phase + omega * dt) % (2.0 * PI);
            if current_phase > PI { current_phase -= 2.0 * PI; }

            predictor.update(current_phase, dt);
        }

        let predicted = predictor.predict_phase(dt);
        let expected = (current_phase + omega * dt) % (2.0 * PI);

        let mut diff = (predicted - expected).abs();
        if diff > PI { diff = 2.0 * PI - diff; }

        assert!(diff < 0.1, "Prediction too far: got {}, expected {}, diff {}", predicted, expected, diff);
    }
}
