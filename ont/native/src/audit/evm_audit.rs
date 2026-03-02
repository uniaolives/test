// ==============================================
// EVM AUDITOR v1.0
// Auditoria contínua do contrato GenesisDAO
// ==============================================

use ethers::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::time::{interval, Duration};
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

abigen!(
    GenesisDAO,
    "genesis/artifacts/GenesisDAO.json",
    event_derives(serde::Deserialize, serde::Serialize)
);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttractorState {
    pub attractor_type: u8,
    pub p: u64,
    pub q: u64,
    pub current_lyapunov: u64,
    pub max_lyapunov: u64,
    pub current_coherence: u64,
    pub min_coherence: u64,
    pub current_energy: u64,
    pub max_energy: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditUpdate {
    pub cycles: u64,
    pub violations: u64,
    pub status: String,
    pub level: String,
    pub message: String,
}

pub struct EVMAuditor<M> {
    contract: GenesisDAO<M>,
    quantum_seed: [u8; 32],
    panic_flag: Arc<AtomicBool>,
    metrics: AuditMetrics,
    tx_broadcast: broadcast::Sender<AuditUpdate>,
}

#[derive(Debug, Clone)]
struct AuditMetrics {
    cycles_completed: u64,
    violations_detected: u64,
    last_healthy_block: u64,
    start_time: std::time::Instant,
}

impl<M: Middleware + 'static> EVMAuditor<M> {
    pub fn new(
        client: Arc<M>,
        contract_address: Address,
        quantum_seed: [u8; 32],
        tx_broadcast: broadcast::Sender<AuditUpdate>,
    ) -> Self {
        let contract = GenesisDAO::new(contract_address, client.clone());

        Self {
            contract,
            quantum_seed,
            panic_flag: Arc::new(AtomicBool::new(false)),
            metrics: AuditMetrics {
                cycles_completed: 0,
                violations_detected: 0,
                last_healthy_block: 0,
                start_time: std::time::Instant::now(),
            },
            tx_broadcast,
        }
    }

    pub async fn start(mut self) {
        log::info!("🚀 Starting EVM Audit Loop for GenesisDAO");

        let mut audit_interval = interval(Duration::from_secs(30));

        // Primeira auditoria imediata
        if let Err(e) = self.perform_full_audit().await {
            log::error!("Initial audit failed: {}", e);
            self.broadcast_status("error", &format!("Initial audit failed: {}", e)).await;
            self.emergency_shutdown().await;
            return;
        }

        // Loop principal
        loop {
            audit_interval.tick().await;

            if self.panic_flag.load(Ordering::SeqCst) {
                log::error!("Panic flag set. Shutting down auditor.");
                break;
            }

            match self.perform_full_audit().await {
                Ok(_) => {
                    self.metrics.cycles_completed += 1;

                    self.broadcast_status("info", "✅ AUDIT CYCLE PASSED").await;

                    // Log a cada 10 ciclos
                    if self.metrics.cycles_completed % 10 == 0 {
                        log::info!(
                            "✅ Audit cycles completed: {}, Violations: {}, Uptime: {:?}",
                            self.metrics.cycles_completed,
                            self.metrics.violations_detected,
                            self.metrics.start_time.elapsed()
                        );
                    }
                }
                Err(fatal) => {
                    log::error!("🔴 Critical audit failure: {}", fatal);
                    self.metrics.violations_detected += 1;

                    self.broadcast_status("error", &format!("🔴 AUDIT FAILURE: {}", fatal)).await;

                    // Protocolo de emergência
                    self.emergency_shutdown().await;
                    break;
                }
            }
        }
    }

    async fn broadcast_status(&self, level: &str, message: &str) {
        let update = AuditUpdate {
            cycles: self.metrics.cycles_completed,
            violations: self.metrics.violations_detected,
            status: level.to_string(),
            level: level.to_string(),
            message: message.to_string(),
        };
        let _ = self.tx_broadcast.send(update);
    }

    async fn perform_full_audit(&mut self) -> Result<(), String> {
        let start_time = std::time::Instant::now();

        // 1. Verificar conexão com a rede
        let block_number = self.get_current_block().await?;

        // 2. Verificar invariante geométrico
        self.check_geometric_invariant().await?;

        // 3. Verificar limites do atrator
        self.check_attractor_bounds().await?;

        // 4. Verificar se contrato ainda responde
        self.check_contract_alive().await?;

        // 5. Atualizar métricas
        self.metrics.last_healthy_block = block_number;

        let elapsed = start_time.elapsed();
        log::debug!("Audit cycle completed in {:?}", elapsed);

        Ok(())
    }

    async fn get_current_block(&self) -> Result<u64, String> {
        self.contract.client()
            .get_block_number()
            .await
            .map(|n| n.as_u64())
            .map_err(|e| format!("Failed to get block number: {}", e))
    }

    async fn check_geometric_invariant(&self) -> Result<(), String> {
        match self.contract.verify_invariant().call().await {
            Ok(true) => Ok(()),
            Ok(false) => Err("Geometric invariant violated (verifyInvariant returned false)".to_string()),
            Err(e) => Err(format!("Failed to call verifyInvariant: {}", e)),
        }
    }

    async fn check_attractor_bounds(&self) -> Result<(), String> {
        let state_result = self.contract.get_attractor_state().call().await;

        match state_result {
            Ok(state) => {
                if state.attractor_type != 0 {
                    return Err(format!("Wrong attractor type: {}", state.attractor_type));
                }

                if state.p != 3.into() || state.q != 5.into() {
                    return Err(format!("Attractor parameters changed: p={}, q={}", state.p, state.q));
                }

                if state.current_lyapunov > state.max_lyapunov {
                    return Err(format!(
                        "Lyapunov exceeded: {} > {}",
                        state.current_lyapunov, state.max_lyapunov
                    ));
                }

                if state.current_coherence < state.min_coherence {
                    return Err(format!(
                        "Coherence below minimum: {} < {}",
                        state.current_coherence, state.min_coherence
                    ));
                }

                if state.current_energy > state.max_energy {
                    return Err(format!(
                        "Energy exceeded: {} > {}",
                        state.current_energy, state.max_energy
                    ));
                }

                Ok(())
            }
            Err(e) => Err(format!("Failed to call getAttractorState: {}", e)),
        }
    }

    async fn check_contract_alive(&self) -> Result<(), String> {
        match self.contract.get_attractor_state().call().await {
            Ok(_) => Ok(()),
            Err(e) => Err(format!("Contract not responding: {}", e)),
        }
    }

    async fn emergency_shutdown(&self) {
        log::error!("🛑 EMERGENCY SHUTDOWN INITIATED");

        // 1. Registrar falha no contrato (se possível)
        let _ = self.contract.report_audit_failure()
            .gas(100000u64)
            .send()
            .await;

        // 2. Log final
        log::error!("🔴 GenesisDAO audit failed after {} cycles",
            self.metrics.cycles_completed);
        log::error!("🕐 Total uptime: {:?}", self.metrics.start_time.elapsed());

        // 3. Aguardar 10 segundos para logs serem flushados
        tokio::time::sleep(Duration::from_secs(10)).await;

        // 4. Terminar processo
        std::process::exit(1);
    }
}

// Implementação do pânico global
pub fn install_panic_hook() {
    std::panic::set_hook(Box::new(|panic_info| {
        log::error!("💀 AUDITOR PANIC: {}", panic_info);

        // Tentar registrar no sistema antes de morrer
        eprintln!("FATAL AUDITOR PANIC: {}", panic_info);

        // Abortar rapidamente
        std::process::abort();
    }));
}
