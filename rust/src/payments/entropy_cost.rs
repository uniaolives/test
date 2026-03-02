// src/payments/entropy_cost.rs

/// Calcula custo de transação baseado em dissipação entrópica
pub struct EntropyCostCalculator {
    /// Coeficiente de conversão: 1 unidade de λ = 0.0001 ETH
    /// Ajustado para compensar custo de cooling do frigorífico dilution
    pub lambda_to_eth_rate: f64,

    /// Taxa base de estabilização (custo fixo por ciclo Schumann)
    pub base_stabilization_fee: u128, // em wei
}

impl EntropyCostCalculator {
    pub fn new() -> Self {
        EntropyCostCalculator {
            lambda_to_eth_rate: 0.0001, // 0.0001 ETH por unidade de Lyapunov
            base_stabilization_fee: 783_000_000_000_000_000, // 0.783 ETH em wei
        }
    }

    /// Custo = Base + (Δλ × Taxa)
    pub fn calculate_transaction_cost(
        &self,
        pre_tx_lyapunov: f64,
        post_tx_lyapunov: f64,
        tx_data_size: usize, // em bytes
    ) -> u128 {
        let delta_lambda = (post_tx_lyapunov - pre_tx_lyapunov).abs();

        // Custo entático: maior para transações que aumentam caos
        let entropy_cost = if post_tx_lyapunov > pre_tx_lyapunov {
            delta_lambda * self.lambda_to_eth_rate * (tx_data_size as f64)
        } else {
            // Transações que reduzem entropia (raras) têm desconto
            -delta_lambda * self.lambda_to_eth_rate * 0.5
        };

        // Converter para wei e adicionar taxa base
        let total_wei = (entropy_cost * 1e18) as i128 + (self.base_stabilization_fee as i128);

        total_wei as u128
    }
}
