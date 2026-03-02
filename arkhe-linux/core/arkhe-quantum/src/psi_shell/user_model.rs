// arkhe-quantum/src/psi_shell/user_model.rs

/// Representa o estado cognitivo do usuário que influencia a ASI.
#[derive(Debug, Clone)]
pub struct UserModel {
    /// Quanto o usuário está "presente" (0.0 = ausente, 1.0 = plenamente focado).
    pub attention: f64,
    /// Última mensagem recebida do usuário (como string).
    pub last_message: Option<String>,
    /// Fator de influência nas ações da ASI (0.0 = nenhuma, 1.0 = máxima).
    pub influence: f64,
}

impl UserModel {
    pub fn new() -> Self {
        UserModel {
            attention: 0.5,
            last_message: None,
            influence: 0.7,
        }
    }

    /// Atualiza o modelo com base em uma mensagem do usuário.
    pub fn process_message(&mut self, msg: &str) {
        self.last_message = Some(msg.to_string());
        if msg.contains("preste atenção") || msg.contains("focus") {
            self.attention = (self.attention + 0.2).min(1.0);
        } else if msg.contains("relaxar") || msg.contains("relax") {
            self.attention = (self.attention - 0.1).max(0.0);
        }
        self.influence = 0.7;
    }

    /// Retorna uma perturbação na matriz densidade baseada no estado do usuário.
    pub fn compute_entropy_perturbation(&self) -> f64 {
        -0.05 * self.attention
    }
}
