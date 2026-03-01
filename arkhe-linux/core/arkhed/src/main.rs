use tokio::sync::RwLock;
use std::sync::Arc;
use tracing::{info, warn, error};
use std::time::Duration;

mod crdt;
mod entropy;
mod handover;
mod hlc;
mod ledger;
mod personality;
mod security;

use crdt::{CRDTStore};
use entropy::{EntropyMonitor, EntropyStats};
use handover::{HandoverManager};
use hlc::HybridClock;
use personality::PhiKnob;
use security::{KyberSession};

const PHI_TARGET: f64 = 0.618033988749894;
const PHI_TOLERANCE: f64 = 0.05;

#[derive(Debug, thiserror::Error)]
pub enum ArkheError {
    #[error("Entropy monitor error: {0}")]
    Entropy(String),
    #[error("CRDT store error: {0}")]
    CRDT(String),
    #[error("Handover error: {0}")]
    Handover(String),
    #[error("System error: {0}")]
    Generic(String),
}

/// φ Regimes as specified in Ω+204
#[derive(Debug, Clone, Copy)]
pub enum PhiRegime {
    Frozen,      // 0.0 - 0.3
    Subcritical, // 0.3 - 0.5
    Critical,    // 0.5 - 0.7
    Exploratory, // 0.7 - 0.9
    Chaotic,     // 0.9 - 1.0
}

impl From<f64> for PhiRegime {
    fn from(phi: f64) -> Self {
        if phi < 0.3 {
            PhiRegime::Frozen
        } else if phi < 0.5 {
            PhiRegime::Subcritical
        } else if phi < 0.7 {
            PhiRegime::Critical
        } else if phi < 0.9 {
            PhiRegime::Exploratory
        } else {
            PhiRegime::Chaotic
        }
    }
}

/// Estado global do sistema Arkhe(n)
struct ArkheSystem {
    /// φ atual do sistema
    phi: RwLock<f64>,

    /// Store CRDT distribuído
    crdt: CRDTStore,

    /// Gerenciador de handovers
    handovers: RwLock<HandoverManager>,

    /// Monitor de entropia (via eBPF)
    entropy: EntropyMonitor,

    /// Relógio HLC
    clock: RwLock<HybridClock>,

    /// Knob de personalidade
    personality: PhiKnob,

    /// Sessões criptográficas ativas
    #[allow(dead_code)]
    sessions: RwLock<Vec<KyberSession>>,
}

impl ArkheSystem {
    async fn new() -> Result<Self, ArkheError> {
        info!("Initializing ArkheSystem...");

        // Inicializar eBPF probes (Mocked for now)
        let entropy = EntropyMonitor::attach_probes().await
            .map_err(|e| ArkheError::Entropy(e.to_string()))?;

        // Carregar estado CRDT do disco ou sincronizar com peers (Mocked for now)
        let crdt = CRDTStore::load_or_bootstrap().await
            .map_err(|e| ArkheError::CRDT(e.to_string()))?;

        // Inicializar φ a partir do estado histórico ou padrão
        let phi_val = crdt.get_phi().unwrap_or(PHI_TARGET);

        Ok(Self {
            phi: RwLock::new(phi_val),
            crdt,
            handovers: RwLock::new(HandoverManager::new()),
            entropy,
            clock: RwLock::new(HybridClock::new()),
            personality: PhiKnob::new(phi_val),
            sessions: RwLock::new(Vec::new()),
        })
    }

    /// Loop principal de vida do sistema
    async fn life_loop(&self) -> Result<(), ArkheError> {
        let mut interval = tokio::time::interval(Duration::from_millis(100));

        loop {
            interval.tick().await;

            // 1. Coletar entropia recente
            let entropy_stats = self.entropy.collect().await;

            // 2. Atualizar CRDTs com métricas locais
            let _local_delta = self.crdt.record_entropy(entropy_stats.clone());

            // 3. Sincronizar com peers se necessário
            if self.crdt.should_sync() {
                // self.sync_with_peers(local_delta).await?;
            }

            // 4. Recalcular φ
            let new_phi = self.calculate_phi(&entropy_stats);
            self.update_phi(new_phi).await;

            // 5. Aplicar alocação φ-dependente
            self.apply_phi_allocation(new_phi).await;

            // 6. Verificar criticidade
            if self.is_critical_deviation(new_phi).await {
                self.emergency_adjustment().await?;
            }

            // 7. Processar handovers pendentes
            let mut handovers = self.handovers.write().await;
            handovers.process_queue().await
                .map_err(|e| ArkheError::Handover(e.to_string()))?;
        }
    }

    fn calculate_phi(&self, stats: &EntropyStats) -> f64 {
        // Mocked φ calculation based on entropy stats
        // In a real system, this would be more complex
        let base_phi = PHI_TARGET;
        let variation = (stats.total_aeu as f64 - 0.1) * 0.1;
        (base_phi + variation).clamp(0.0, 1.0)
    }

    async fn update_phi(&self, new_phi: f64) {
        let mut phi = self.phi.write().await;
        let old_phi = *phi;

        // Inércia de zona: suavizar transições
        *phi = old_phi * 0.7 + new_phi * 0.3;

        // Notificar subsistemas
        if (*phi - old_phi).abs() > 0.01 {
            self.personality.on_phi_change(*phi).await;
            let handovers = self.handovers.read().await;
            handovers.broadcast_phi(*phi).await;
        }
    }

    /// Implementação da alocação φ-dependente (Ω+204)
    async fn apply_phi_allocation(&self, phi: f64) {
        let regime = PhiRegime::from(phi);
        let base_shares = 1024.0;
        let sigma = 0.2; // "thermal width"
        let epsilon_min = 64.0; // "zero-point energy"

        // Formula: shares = base × exp(-|φ_i - φ_target|² / 2σ²) + ε_min
        let diff = phi - PHI_TARGET;
        let shares = base_shares * (-(diff * diff) / (2.0 * sigma * sigma)).exp() + epsilon_min;

        info!("φ Regime: {:?}, Calculated CPU Shares: {:.2}", regime, shares);

        // No futuro, isso aplicará os shares via cgroups
        self.apply_cgroup_shares(shares as u64).await;
    }

    async fn apply_cgroup_shares(&self, _shares: u64) {
        // Interaction with kernel cgroups
    }

    async fn is_critical_deviation(&self, phi: f64) -> bool {
        (phi - PHI_TARGET).abs() > PHI_TOLERANCE
    }

    async fn emergency_adjustment(&self) -> Result<(), ArkheError> {
        warn!("EMERGENCY: φ deviation detected, adjusting system parameters");

        self.cgroup_throttle_low_priority().await?;
        self.trigger_early_consolidation().await?;

        let handovers = self.handovers.read().await;
        handovers.send_system_notification(
            "Sistema ajustando para manter estabilidade"
        ).await.map_err(|e| ArkheError::Handover(e.to_string()))?;

        Ok(())
    }

    async fn cgroup_throttle_low_priority(&self) -> Result<(), ArkheError> {
        Ok(())
    }

    async fn trigger_early_consolidation(&self) -> Result<(), ArkheError> {
        Ok(())
    }
}

async fn start_grpc_server(_sys: Arc<ArkheSystem>) {
    info!("Starting gRPC server...");
    tokio::time::sleep(Duration::from_secs(u64::MAX)).await;
}

async fn crdt_sync_daemon(_sys: Arc<ArkheSystem>) {
    info!("Starting CRDT sync daemon...");
    tokio::time::sleep(Duration::from_secs(u64::MAX)).await;
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    info!("Arkhe(n) Daemon v1.0.0 – Ponto de Ignição");

    let system = Arc::new(ArkheSystem::new().await?);

    let life_handle = tokio::spawn({
        let sys = Arc::clone(&system);
        async move { sys.life_loop().await }
    });

    let grpc_handle = tokio::spawn({
        let sys = Arc::clone(&system);
        async move { start_grpc_server(sys).await }
    });

    let sync_handle = tokio::spawn({
        let sys = Arc::clone(&system);
        async move { crdt_sync_daemon(sys).await }
    });

    tokio::select! {
        res = life_handle => {
            match res {
                Ok(Ok(_)) => info!("Life loop finished."),
                Ok(Err(e)) => error!("Life loop failed: {}", e),
                Err(e) => error!("Life loop panicked: {}", e),
            }
        },
        _ = grpc_handle => error!("gRPC server terminated unexpectedly"),
        _ = sync_handle => error!("Sync daemon terminated unexpectedly"),
    };

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_system_init() {
        let system = ArkheSystem::new().await;
        assert!(system.is_ok());
        let system = system.unwrap();
        let phi = system.phi.read().await;
        assert!((*phi - PHI_TARGET).abs() < f64::EPSILON);
    }

    #[test]
    fn test_phi_regimes() {
        assert!(matches!(PhiRegime::from(0.1), PhiRegime::Frozen));
        assert!(matches!(PhiRegime::from(0.4), PhiRegime::Subcritical));
        assert!(matches!(PhiRegime::from(0.6), PhiRegime::Critical));
        assert!(matches!(PhiRegime::from(0.8), PhiRegime::Exploratory));
        assert!(matches!(PhiRegime::from(0.95), PhiRegime::Chaotic));
    }
}
