//! arkhe_diplomatic::adaptive_kalman
//! Filtro de Kalman Adaptativo de 3ª Ordem.
//! Modula a covariância de medição (R) baseado na coerência termodinâmica (C_local).

pub struct AdaptiveKalmanPredictor {
    // Vetor de Estado: [phi, omega, alpha]
    pub state: [f64; 3],
    // Matriz de Covariância do Erro (P)
    pub p: [[f64; 3]; 3],
    // Ruído do Processo (Q) - Incerteza do modelo físico
    pub q: f64,
    // Ruído de Medição Base (R_base) - Ruído térmico ideal do SDR
    pub r_base: f64,
    // Fator de Adaptação (β) - Agressividade do isolamento acústico
    pub adaptation_rate: f64,
}

impl AdaptiveKalmanPredictor {
    pub fn new(q: f64, r_base: f64, adaptation_rate: f64) -> Self {
        Self {
            state: [0.0, 0.0, 0.0],
            p: [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            q,
            r_base,
            adaptation_rate, // Ex: 10.0 a 50.0 para penalizar severamente a baixa coerência
        }
    }

    /// Atualiza o filtro com a medição de fase e a coerência do SDR (0.0 a 1.0)
    pub fn update(&mut self, measured_phase: f64, dt: f64, coherence: f64) {
        // 1. Modulação Dinâmica de R (A Magia Adaptativa)
        // Se coherence = 1.0, R_dynamic = R_base
        // Se coherence cai para 0.2, R explode exponencialmente, zerando o ganho K
        let penalty = 1.0 - coherence.clamp(0.01, 1.0);
        let r_dynamic = self.r_base * (1.0 + self.adaptation_rate * penalty.exp());

        // 2. Matriz de Transição de Estado (F) baseada na inércia
        let f_matrix = [
            [1.0, dt, 0.5 * dt * dt],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0],
        ];

        // 3. Predição do Estado (Priori)
        let mut next_state = [0.0; 3];
        for i in 0..3 {
            for j in 0..3 {
                next_state[i] += f_matrix[i][j] * self.state[j];
            }
        }

        next_state[0] = next_state[0].rem_euclid(2.0 * std::f64::consts::PI);
        if next_state[0] > std::f64::consts::PI {
            next_state[0] -= 2.0 * std::f64::consts::PI;
        }

        // 4. Atualização da Covariância Priori
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

        // 5. Inovação
        let mut y = measured_phase - next_state[0];
        while y > std::f64::consts::PI { y -= 2.0 * std::f64::consts::PI; }
        while y < -std::f64::consts::PI { y += 2.0 * std::f64::consts::PI; }

        // 6. Ganho de Kalman Adaptativo
        // Aqui o r_dynamic entra em ação. Se r_dynamic for colossal, S fica colossal,
        // e k_gain tende a zero. O filtro ignora a medição e usa apenas a inércia.
        let s = p_next[0][0] + r_dynamic;
        let k_gain = [p_next[0][0] / s, p_next[1][0] / s, p_next[2][0] / s];

        // 7. Atualização do Estado Posteriori
        for i in 0..3 {
            self.state[i] = next_state[i] + k_gain[i] * y;
        }

        // 8. Atualização da Covariância Posteriori
        for i in 0..3 {
            for j in 0..3 {
                self.p[i][j] = p_next[i][j] - k_gain[i] * p_next[0][j];
            }
        }
    }

    pub fn predict_phase(&self, target_dt: f64) -> f64 {
        let predicted = self.state[0]
            + self.state[1] * target_dt
            + 0.5 * self.state[2] * target_dt * target_dt;

        let mut normalized = predicted.rem_euclid(2.0 * std::f64::consts::PI);
        if normalized > std::f64::consts::PI {
            normalized -= 2.0 * std::f64::consts::PI;
        }
        normalized
    }
}
