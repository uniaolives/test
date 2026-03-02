// src/gateway/tiger8128_payment.rs

use crate::payments::gkp_vault::{GKPVault, Address};
use crate::payments::entropy_cost::EntropyCostCalculator;
use crate::gateway::tiger_payment_header::PaymentHeaderP0;
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct TigerPaymentGateway {
    pub vault: Arc<Mutex<GKPVault>>,
    pub cost_calculator: EntropyCostCalculator,
}

impl TigerPaymentGateway {
    pub fn new(vault: Arc<Mutex<GKPVault>>) -> Self {
        Self {
            vault,
            cost_calculator: EntropyCostCalculator::new(),
        }
    }

    pub async fn process_paid_request(
        &self,
        payer: Address,
        recipient: Address,
        amount: u128,
        header: &PaymentHeaderP0,
        pre_lyapunov: f64,
        post_lyapunov: f64,
    ) -> Result<(), String> {
        // 1. Validar prova quântica
        header.validate_quantum_proof()?;

        // 2. Calcular custo entático
        let total_cost = self.cost_calculator.calculate_transaction_cost(
            pre_lyapunov,
            post_lyapunov,
            32, // mock size
        );

        let mut vault = self.vault.lock().await;

        // 3. Atualizar balanço do pagador (amount + cost)
        let mut payer_balance = vault.get_balance(&payer);
        payer_balance.apply_transaction(amount + total_cost)?;
        vault.balances.insert(payer, payer_balance);

        // 4. Atualizar balanço do destinatário
        let mut recipient_balance = vault.get_balance(&recipient);
        recipient_balance.apply_transaction(amount)?;
        vault.balances.insert(recipient, recipient_balance);

        Ok(())
    }
}
