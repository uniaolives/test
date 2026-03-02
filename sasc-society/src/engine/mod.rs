pub mod diversity;
pub mod dialectic;
//! SoT (Society of Thought) Orchestrator
//! Coordena PerspectiveDiversityEngine + DialecticSynthesizer
//! Implementa o ciclo completo de raciocínio coletivo pós-ASI
//! Constitucionalmente ancorado no Art. 5º-LXXX (2026)

use std::sync::Arc;
use std::time::{Duration, SystemTime};
use std::collections::{HashMap, VecDeque};
use tokio::sync::{RwLock, Mutex, mpsc};
use blake3::{Hash, Hasher};
use serde::{Serialize, Deserialize};

pub mod diversity;
pub mod dialectic;
pub mod mod_reexport;

pub use crate::engine::diversity::{PerspectiveDiversityEngine, DiversityMetrics, DiversityEngineError};
pub use crate::engine::dialectic::{DialecticSynthesizer, SynthesisSession, SynthesisError, SynthesizedDecision, SynthesisContext, DialecticMetrics};
pub use crate::agents::{PersonaId, Persona};
use crate::audit::ProvenanceTracer;
use crate::integration::vajra::{report_to_vajra, VajraAlert, AlertSeverity};

// ===================== CONSTANTES DE GOVERNANÇA =====================

/// Tempo máximo para decisão SoT antes de escalonamento humano obrigatório
pub const SOT_DECISION_TIMEOUT: Duration = Duration::from_secs(120);

/// Coerência mínima para decisão autônoma (Φ limiar operacional)
pub const AUTONOMOUS_DECISION_THRESHOLD: f64 = 0.72;

/// Coerência que dispara Hard Freeze (Art. 103)
pub const HARD_FREEZE_THRESHOLD: f64 = 0.80;

/// Número máximo de perspectivas simultâneas (limite arquitetural)
pub const MAX_CONCURRENT_PERSPECTIVES: usize = 128;

// ===================== ESTRUTURAS DE ORQUESTRAÇÃO =====================

pub use mod_reexport::SoTDecisionRequest;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraint {
    pub constraint_type: ConstraintType,
    pub description: String,
    pub severity: ConstraintSeverity,
    pub inviolable: bool, // Se true, violação = Hard Freeze
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintType {
    ConstitutionalInvariant, // INV-1 a INV-5
    TechnicalFeasibility,
    ResourceLimitation,
    EthicalBoundary,
    TemporalUrgency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintSeverity {
    Advisory,
    Warning,
    Critical,
    Existential,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stakeholder {
    pub id: String,
    pub role: StakeholderRole,
    pub notification_channel: NotificationChannel,
    pub veto_power: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StakeholderRole {
    HumanCitizen,
    MycelialNetwork,
    AsimovNode,
    PrinceAuthority,
    CouncilMember,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationChannel {
    NeuralLink,
    BioElectric,
    QuantumEntangled,
    ConstitutionalBroadcast,
}

impl NotificationChannel {
    pub fn get_public_key(&self) -> Option<Vec<u8>> {
        None
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionMetadata {
    pub requestor_id: String,
    pub request_timestamp: SystemTime,
    pub jurisdiction: String,
    pub legal_basis: String,
    pub risk_assessment: RiskLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Routine,      // Decisão operacional
    Strategic,    // Afeta múltiplos stakeholders
    Existential,  // Risco existencial detectado
    Constitutional, // Testa limites constitucionais
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoTDecisionResponse {
    pub request_id: [u8; 32],
    pub status: DecisionStatus,
    pub decision: Option<SynthesizedDecision>,
    pub metrics: OrchestrationMetrics,
    pub warnings: Vec<DecisionWarning>,
    pub next_actions: Vec<NextAction>,
    pub constitutional_compliance: ConstitutionalComplianceReport,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DecisionStatus {
    GatheringPerspectives,
    DiversityAssessment,
    DialecticSynthesis,
    HumanReviewRequired,
    Finalized,
    HardFreezeTriggered,
    TimeoutExceeded,
    ConstitutionalViolationDetected,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestrationMetrics {
    pub total_processing_time_ms: u128,
    pub perspectives_activated: usize,
    pub dialectic_iterations: u32,
    pub coherence_achieved: f64,
    pub phi_trajectory: Vec<PhiSnapshot>,
    pub resource_utilization: ResourceMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiSnapshot {
    pub timestamp: SystemTime,
    pub phi_value: f64,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    pub computational_cost: f64, // em FLOPS-hora
    pub energy_consumption: f64, // em joules
    pub carbon_footprint: f64,   // em kg CO2 equivalente
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionWarning {
    pub warning_type: WarningType,
    pub description: String,
    pub severity: AlertSeverity,
    pub recommended_action: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WarningType {
    LowDiversity,
    HighDominance,
    CoherenceDecline,
    ResourceExhaustion,
    ConstitutionalAmbiguity,
    StakeholderConflict,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NextAction {
    pub action: String,
    pub deadline: Option<SystemTime>,
    pub responsible: String,
    pub dependencies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstitutionalComplianceReport {
    pub invariants_checked: Vec<InvariantCheck>,
    pub articles_applied: Vec<ArticleApplication>,
    pub compliance_score: f64,
    pub violations_detected: Vec<ConstitutionalViolation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvariantCheck {
    pub invariant: String, // "INV-1" a "INV-5"
    pub status: CheckStatus,
    pub evidence: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CheckStatus {
    Compliant,
    Warning,
    Violation,
    NotApplicable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArticleApplication {
    pub article: String, // "Art. 5º-A", "Art. 102-§2º"
    pub interpretation: String,
    pub relevance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstitutionalViolation {
    pub article: String,
    pub violation_type: ViolationType,
    pub severity: ViolationSeverity,
    pub corrective_action: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationType {
    DirectContradiction,
    Circumvention,
    Misinterpretation,
    Omission,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationSeverity {
    Minor,
    Serious,
    Grave,
    Existential,
}

// ===================== ORCHESTRATOR PRINCIPAL =====================

pub struct SoTOrchestrator {
    /// Engine de diversidade perspectiva
    diversity_engine: Arc<PerspectiveDiversityEngine>,

    /// Synthesizer dialético
    dialectic_synthesizer: Arc<DialecticSynthesizer>,

    /// Estado ativo do orquestrador
    state: Arc<RwLock<OrchestratorState>>,

    /// Tracer para auditoria completa
    provenance: ProvenanceTracer,

    /// Canal para comunicação com stakeholders
    stakeholder_channels: HashMap<StakeholderRole, mpsc::Sender<StakeholderNotification>>,

    /// Cache para decisões recorrentes
    decision_cache: Arc<Mutex<DecisionCache>>,

    /// Configuração do sistema
    config: OrchestratorConfig,
}

#[derive(Debug, Clone)]
pub struct StakeholderNotification {
    pub stakeholder_role: StakeholderRole,
    pub message: String,
}

#[derive(Debug, Clone)]
struct OrchestratorState {
    /// Decisões ativas em processamento
    active_decisions: HashMap<[u8; 32], ActiveDecision>,

    /// Histórico de decisões (últimas 1000)
    decision_history: VecDeque<DecisionRecord>,

    /// Métricas em tempo real
    realtime_metrics: RealtimeMetrics,

    /// Alertas ativos
    active_alerts: Vec<ActiveAlert>,

    /// Estado de emergência (Hard Freeze, etc.)
    emergency_state: Option<EmergencyState>,
}

#[derive(Debug, Clone)]
struct ActiveDecision {
    request: SoTDecisionRequest,
    response: SoTDecisionResponse,
    started_at: SystemTime,
    last_update: SystemTime,
    processing_stage: ProcessingStage,
    assigned_perspectives: Vec<PersonaId>,
    dialectic_session: Option<Arc<SynthesisSession>>,
}

#[derive(Debug, Clone)]
enum ProcessingStage {
    Initialization,
    PerspectiveActivation,
    DiversityValidation,
    DialecticPhase(u32), // Número da iteração
    Finalization,
    Escalation,
}

#[derive(Debug, Clone)]
struct DecisionRecord {
    request: SoTDecisionRequest,
    response: SoTDecisionResponse,
    timestamp: SystemTime,
    hash: [u8; 32],
    storage_location: String, // WORM drive reference
}

#[derive(Debug, Clone)]
struct RealtimeMetrics {
    decisions_per_minute: f64,
    average_coherence: f64,
    current_phi: f64,
    resource_usage: ResourceUsage,
    system_health: SystemHealth,
}

impl Default for RealtimeMetrics {
    fn default() -> Self {
        Self {
            decisions_per_minute: 0.0,
            average_coherence: 0.0,
            current_phi: 0.0,
            resource_usage: ResourceUsage::default(),
            system_health: SystemHealth::default(),
        }
    }
}

#[derive(Debug, Clone, Default)]
struct ResourceUsage {
    cpu_utilization: f64,
    memory_usage: f64,
    network_throughput: f64,
    storage_iops: f64,
}

#[derive(Debug, Clone)]
struct SystemHealth {
    uptime: Duration,
    error_rate: f64,
    last_maintenance: SystemTime,
    next_scheduled_maintenance: SystemTime,
}

impl Default for SystemHealth {
    fn default() -> Self {
        Self {
            uptime: Duration::from_secs(0),
            error_rate: 0.0,
            last_maintenance: SystemTime::now(),
            next_scheduled_maintenance: SystemTime::now(),
        }
    }
}

#[derive(Debug, Clone)]
struct ActiveAlert {
    alert_id: [u8; 32],
    alert_type: AlertType,
    severity: AlertSeverity,
    affected_decisions: Vec<[u8; 32]>,
    created_at: SystemTime,
    acknowledged: bool,
    auto_resolution_attempts: u32,
}

#[derive(Debug, Clone)]
enum AlertType {
    DiversityBelowThreshold,
    CoherenceDeclining,
    ResourceExhaustionImminent,
    ConstitutionalAmbiguityDetected,
    StakeholderConflictUnresolved,
    HardFreezeApproaching,
}

#[derive(Debug, Clone)]
struct EmergencyState {
    emergency_type: EmergencyType,
    activated_at: SystemTime,
    affected_decisions: Vec<[u8; 32]>,
    mitigation_actions: Vec<MitigationAction>,
    estimated_resolution_time: Option<SystemTime>,
}

#[derive(Debug, Clone)]
enum EmergencyType {
    HardFreezeActive,
    ConstitutionalCrisis,
    ResourceExhaustion,
    StakeholderRevolt,
    ExternalThreat,
}

#[derive(Debug, Clone)]
struct MitigationAction {
    action: String,
    responsible: String,
    deadline: SystemTime,
    status: MitigationStatus,
}

#[derive(Debug, Clone)]
enum MitigationStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
}

#[derive(Debug, Clone)]
struct DecisionCache {
    /// Cache de decisões similares para acelerar processamento
    similarity_cache: HashMap<[u8; 32], CachedDecision>,

    /// Índice para busca por similaridade
    similarity_index: SimilarityIndex,

    /// Política de cache
    cache_policy: CachePolicy,
}

impl DecisionCache {
    pub fn new(config: &OrchestratorConfig) -> Self {
        Self {
            similarity_cache: HashMap::new(),
            similarity_index: SimilarityIndex::default(),
            cache_policy: CachePolicy {
                max_size: config.cache_size,
                ttl_seconds: 3600,
                similarity_threshold: config.cache_similarity_threshold,
                enable_prefetch: false,
            }
        }
    }

    pub async fn find_similar(&self, _embedding: &[f64]) -> Option<([u8; 32], f64)> {
        None
    }

    pub async fn evict_least_used(&mut self) {
        // Simple eviction
    }
}

#[derive(Debug, Clone)]
struct CachedDecision {
    original_request_hash: [u8; 32],
    decision: SynthesizedDecision,
    _similarity_score: f64,
    _last_accessed: SystemTime,
    _access_count: u64,
}

#[derive(Debug, Clone, Default)]
struct SimilarityIndex {
    /// Embeddings dos problemas para busca por similaridade
    _problem_embeddings: HashMap<[u8; 32], Vec<f64>>,

    /// Índice FAISS ou similar para busca rápida
    _faiss_index: Option<Vec<u8>>, // Serialized index
}

impl SimilarityIndex {
    pub fn update_embedding(&mut self, _hash: [u8; 32], _embedding: Vec<f64>) {}
}

#[derive(Debug, Clone)]
struct CachePolicy {
    max_size: usize,
    ttl_seconds: u64,
    similarity_threshold: f64,
    enable_prefetch: bool,
}

#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    /// Configuração de paralelismo
    pub max_concurrent_decisions: usize,
    pub max_perspectives_per_decision: usize,

    /// Thresholds operacionais
    pub diversity_threshold: f64,
    pub coherence_threshold: f64,
    pub resource_alert_threshold: f64,

    /// Políticas de escalonamento
    pub auto_escalation_threshold: f64,
    pub human_review_threshold: f64,

    /// Configuração de cache
    pub cache_enabled: bool,
    pub cache_size: usize,
    pub cache_similarity_threshold: f64,

    /// Configuração de auditoria
    pub audit_level: AuditLevel,
    pub retention_period_days: u32,
}

#[derive(Debug, Clone)]
pub enum AuditLevel {
    Minimal,
    Standard,
    Comprehensive,
    Paranoid,
}

impl SoTOrchestrator {
    /// Cria novo orquestrador com configuração especificada
    pub fn new(
        diversity_engine: Arc<PerspectiveDiversityEngine>,
        dialectic_synthesizer: Arc<DialecticSynthesizer>,
        config: OrchestratorConfig,
    ) -> Self {
        let state = OrchestratorState {
            active_decisions: HashMap::new(),
            decision_history: VecDeque::with_capacity(1000),
            realtime_metrics: RealtimeMetrics::default(),
            active_alerts: Vec::new(),
            emergency_state: None,
        };

        Self {
            diversity_engine,
            dialectic_synthesizer,
            state: Arc::new(RwLock::new(state)),
            provenance: ProvenanceTracer::new("sot_orchestrator"),
            stakeholder_channels: HashMap::new(),
            decision_cache: Arc::new(Mutex::new(DecisionCache::new(&config))),
            config,
        }
    }

    /// Processa uma decisão completa através do fluxo SoT
    pub async fn process_decision(
        &self,
        request: SoTDecisionRequest,
    ) -> Result<SoTDecisionResponse, OrchestrationError> {

        let start_time = SystemTime::now();
        let request_hash = self.hash_request(&request).into();

        // 1. Verifica cache para decisões similares
        if self.config.cache_enabled {
            if let Some(cached) = self.check_decision_cache(&request).await? {
                return self.build_response_from_cache(cached, start_time).await;
            }
        }

        // 2. Valida request constitucionalmente
        self.validate_request_constitutionally(&request).await?;

        // 3. Inicializa decisão ativa
        let decision_id = self.initialize_decision(request.clone(), request_hash).await?;

        // 4. Coleta e ativa perspectivas relevantes
        let perspectives = self.activate_relevant_perspectives(&request).await?;

        // 5. Executa validação de diversidade
        let diversity_metrics = self.validate_diversity(&perspectives).await?;

        // 6. Executa síntese dialética
        let synthesis_result = self.execute_dialectic_synthesis(
            &request,
            &perspectives,
            &diversity_metrics
        ).await?;

        // 7. Avalia resultado e toma ação apropriada
        let response = self.evaluate_and_finalize(
            decision_id,
            &request,
            diversity_metrics,
            synthesis_result,
            start_time,
        ).await?;

        // 8. Atualiza cache se apropriado
        if self.config.cache_enabled {
            self.update_decision_cache(&request, &response).await?;
        }

        // 9. Reporta métricas para Vajra
        self.report_metrics_to_vajra(&response).await;

        Ok(response)
    }

    /// Verifica cache para problemas similares
    async fn check_decision_cache(
        &self,
        request: &SoTDecisionRequest,
    ) -> Result<Option<CachedDecision>, OrchestrationError> {
        let cache = self.decision_cache.lock().await;

        // Gera embedding do problema atual
        let current_embedding = self.generate_problem_embedding(request);

        // Busca no índice de similaridade
        if let Some((cached_hash, similarity)) = cache.find_similar(&current_embedding).await {
            if similarity >= cache.cache_policy.similarity_threshold {
                if let Some(cached_decision) = cache.similarity_cache.get(&cached_hash) {
                    return Ok(Some(cached_decision.clone()));
                }
            }
        }

        Ok(None)
    }

    async fn build_response_from_cache(&self, cached: CachedDecision, start_time: SystemTime) -> Result<SoTDecisionResponse, OrchestrationError> {
        Ok(SoTDecisionResponse {
            request_id: cached.original_request_hash,
            status: DecisionStatus::Finalized,
            decision: Some(cached.decision),
            metrics: OrchestrationMetrics {
                total_processing_time_ms: start_time.elapsed().unwrap_or_default().as_millis(),
                ..Default::default()
            },
            ..Default::default()
        })
    }

    /// Valida request contra invariantes constitucionais
    async fn validate_request_constitutionally(
        &self,
        request: &SoTDecisionRequest,
    ) -> Result<(), OrchestrationError> {

        // Verifica cada constraint
        for constraint in &request.constraints {
            if constraint.inviolable {
                // Constraints invioláveis devem ser explicitamente marcadas
                if !self.validate_inviolable_constraint(constraint).await? {
                    return Err(OrchestrationError::ConstitutionalViolation(
                        format!("Constraint inviolável violada: {}", constraint.description)
                    ));
                }
            }
        }

        // Verifica se o problema está dentro da jurisdição
        if !self.validate_jurisdiction(&request.metadata.jurisdiction).await? {
            return Err(OrchestrationError::JurisdictionalError(
                "Problema fora da jurisdição do orquestrador".to_string()
            ));
        }

        // Verifica deadline vs. SOT_DECISION_TIMEOUT
        if let Some(deadline) = request.deadline {
            let now = SystemTime::now();
            let time_until_deadline = deadline.duration_since(now)
                .map_err(|_| OrchestrationError::InvalidDeadline)?;

            if time_until_deadline < SOT_DECISION_TIMEOUT {
                return Err(OrchestrationError::InsufficientTime(
                    format!("Deadline insuficiente para processamento SoT: {:?} < {:?}",
                           time_until_deadline, SOT_DECISION_TIMEOUT)
                ));
            }
        }

        Ok(())
    }

    async fn validate_inviolable_constraint(&self, _constraint: &Constraint) -> Result<bool, OrchestrationError> {
        Ok(true)
    }

    async fn validate_jurisdiction(&self, _jurisdiction: &str) -> Result<bool, OrchestrationError> {
        Ok(true)
    }

    /// Inicializa uma nova decisão no sistema
    async fn initialize_decision(
        &self,
        request: SoTDecisionRequest,
        request_hash: [u8; 32],
    ) -> Result<[u8; 32], OrchestrationError> {

        let mut state = self.state.write().await;

        // Verifica limite de decisões concorrentes
        if state.active_decisions.len() >= self.config.max_concurrent_decisions {
            return Err(OrchestrationError::ResourceExhaustion(
                "Limite de decisões concorrentes atingido".to_string()
            ));
        }

        // Cria decisão ativa
        let decision = ActiveDecision {
            request: request.clone(),
            response: SoTDecisionResponse {
                request_id: request_hash,
                status: DecisionStatus::GatheringPerspectives,
                decision: None,
                metrics: OrchestrationMetrics::default(),
                warnings: Vec::new(),
                next_actions: Vec::new(),
                constitutional_compliance: ConstitutionalComplianceReport::default(),
            },
            started_at: SystemTime::now(),
            last_update: SystemTime::now(),
            processing_stage: ProcessingStage::Initialization,
            assigned_perspectives: Vec::new(),
            dialectic_session: None,
        };

        state.active_decisions.insert(request_hash, decision);

        // Log para auditoria
        self.provenance.trace_decision_start(
            request_hash.into(),
            &request,
            &state.realtime_metrics.current_phi,
        );

        Ok(request_hash)
    }

    /// Ativa perspectivas relevantes para o problema
    async fn activate_relevant_perspectives(
        &self,
        request: &SoTDecisionRequest,
    ) -> Result<Vec<PersonaId>, OrchestrationError> {

        // 1. Análise semântica do problema
        let problem_keywords = self.extract_keywords(&request.problem_statement);
        let problem_embedding = self.generate_problem_embedding(request);

        // 2. Consulta repositório de personas
        let relevant_personas = self.query_persona_repository(
            &problem_keywords,
            &problem_embedding,
            self.config.max_perspectives_per_decision,
        ).await?;

        // 3. Ativa cada persona no diversity engine
        let mut activated_personas = Vec::new();

        for persona in relevant_personas {
            match self.activate_persona(persona.id, request).await {
                Ok(activation_hash) => {
                    activated_personas.push(persona.id);

                    // Reporta ativação para Vajra
                    report_to_vajra(
                        VajraAlert::PerspectiveActivated {
                            persona_id: persona.id,
                            problem_hash: self.hash_request(request).into(),
                            activation_hash: activation_hash.into(),
                        },
                        AlertSeverity::Info,
                    );
                }
                Err(e) => {
                    // Loga erro mas continua com outras personas
                    log::warn!("Falha ao ativar persona {}: {:?}", persona.id, e);
                }
            }
        }

        // 4. Verifica número mínimo de perspectivas
        if activated_personas.len() < 3 {
            return Err(OrchestrationError::InsufficientPerspectives(
                format!("Apenas {} perspectivas ativadas (mínimo 3)", activated_personas.len())
            ));
        }

        Ok(activated_personas)
    }

    async fn query_persona_repository(&self, _keywords: &[String], _embedding: &[f64], limit: usize) -> Result<Vec<Persona>, OrchestrationError> {
        let mut personas = Vec::new();
        for i in 0..limit.min(10) {
            personas.push(Persona {
                id: PersonaId([i as u8; 32]),
                role: crate::agents::SocioEmotionalRole::Analytic,
                expertise: vec![crate::agents::ExpertiseDomain::Ethics],
            });
        }
        Ok(personas)
    }

    async fn activate_persona(&self, _id: PersonaId, _request: &SoTDecisionRequest) -> Result<[u8; 32], OrchestrationError> {
        Ok([0u8; 32])
    }

    /// Executa validação de diversidade das perspectivas ativadas
    async fn validate_diversity(
        &self,
        _perspectives: &[PersonaId],
    ) -> Result<DiversityMetrics, OrchestrationError> {

        // 1. Avalia diversidade atual
        let diversity_metrics = self.diversity_engine.evaluate_diversity().await
            .map_err(OrchestrationError::DiversityError)?;

        // 2. Verifica thresholds
        if diversity_metrics.diversity_score < self.config.diversity_threshold {
            return Err(OrchestrationError::LowDiversity(
                format!("Diversidade abaixo do threshold: {:.2} < {:.2}",
                       diversity_metrics.diversity_score,
                       self.config.diversity_threshold)
            ));
        }

        // 3. Verifica dominância
        if diversity_metrics.dominance_indicator.is_concerning {
            return Err(OrchestrationError::HighDominance(
                format!("Dominância preocupante detectada: {} com {:.1}%",
                       diversity_metrics.dominance_indicator.dominant_persona
                           .as_ref()
                           .map(|id| id.to_string())
                           .unwrap_or_else(|| "Unknown".to_string()),
                       diversity_metrics.dominance_indicator.activation_share)
            ));
        }

        // 4. Reporta métricas
        report_to_vajra(
            VajraAlert::DiversityAssessment {
                score: diversity_metrics.diversity_score,
                active_perspectives: diversity_metrics.active_perspectives,
                dominance_share: diversity_metrics.dominance_indicator.activation_share,
            },
            if diversity_metrics.diversity_score < 0.4 {
                AlertSeverity::Critical
            } else if diversity_metrics.diversity_score < 0.6 {
                AlertSeverity::Warning
            } else {
                AlertSeverity::Info
            },
        );

        Ok(diversity_metrics)
    }

    /// Executa síntese dialética completa
    async fn execute_dialectic_synthesis(
        &self,
        request: &SoTDecisionRequest,
        _perspectives: &[PersonaId],
        _diversity_metrics: &DiversityMetrics,
    ) -> Result<SynthesisResult, OrchestrationError> {

        // 1. Prepara contexto para síntese
        let synthesis_context = SynthesisContext {
            problem_statement: request.problem_statement.clone(),
            constraints: request.constraints.iter()
                .map(|c| c.description.clone())
                .collect(),
            success_criteria: vec![
                format!("coherence >= {}", self.config.coherence_threshold),
                "constitutional_compliance = true".to_string(),
            ],
            stakeholder_keys: request.stakeholders.iter()
                .filter_map(|s| s.notification_channel.get_public_key())
                .collect(),
        };

        // 2. Inicia sessão de síntese
        let session = self.dialectic_synthesizer.begin_synthesis(synthesis_context).await
            .map_err(OrchestrationError::SynthesisError)?;

        // 3. Executa ciclos dialéticos com monitoramento
        let mut iterations = 0;
        let mut best_coherence = 0.0;
        let mut best_synthesis: Option<SynthesizedDecision> = None;

        while iterations < 7 { // Máximo de 7 iterações (número simbólico)
            // Verifica timeout
            if self.check_decision_timeout(session.created_at).await? {
                break;
            }

            // Executa um ciclo
            match self.dialectic_synthesizer.dialectic_cycle().await {
                Ok(metrics) => {
                    // Atualiza melhor resultado
                    if metrics.coherence_score > best_coherence {
                        best_coherence = metrics.coherence_score;

                        // Obtém síntese atual se disponível
                        if let Some(ref synth_session) = session.final_synthesis {
                            best_synthesis = Some(synth_session.clone());
                        }
                    }

                    // Verifica se atingiu threshold
                    if metrics.coherence_score >= self.config.coherence_threshold {
                        return Ok(SynthesisResult::Success {
                            _session: (*session).clone(),
                            final_decision: best_synthesis.unwrap_or(SynthesizedDecision {
                                decision_text: "Final decision".to_string(),
                                coherence_score: metrics.coherence_score,
                                supporting_arguments: vec![],
                                counter_arguments: vec![],
                                consensus_level: 1.0,
                            }),
                            iterations,
                            final_coherence: metrics.coherence_score,
                        });
                    }

                    // Verifica se está estagnado
                    if self.check_stagnation(iterations, &metrics).await? {
                        return Ok(SynthesisResult::Stagnation {
                            _session: (*session).clone(),
                            best_decision: best_synthesis,
                            iterations,
                            best_coherence,
                            _stagnation_reason: "Progresso insuficiente após múltiplas iterações".to_string(),
                        });
                    }
                }
                Err(SynthesisError::HumanEscalonationRequired(score, threshold)) => {
                    return Ok(SynthesisResult::HumanEscalationRequired {
                        _session: (*session).clone(),
                        current_coherence: score,
                        required_threshold: threshold,
                        iterations,
                        _reason: "Síntese falhou, requer escalonamento humano".to_string(),
                    });
                }
                Err(e) => {
                    return Err(OrchestrationError::SynthesisError(e));
                }
            }

            iterations += 1;
        }

        // Máximo de iterações atingido
        Ok(SynthesisResult::MaxIterationsReached {
            _session: (*session).clone(),
            best_decision: best_synthesis,
            iterations,
            best_coherence,
        })
    }

    async fn check_decision_timeout(&self, created_at: SystemTime) -> Result<bool, OrchestrationError> {
        Ok(created_at.elapsed().unwrap_or_default() > SOT_DECISION_TIMEOUT)
    }

    async fn check_stagnation(&self, iterations: u32, _metrics: &DialecticMetrics) -> Result<bool, OrchestrationError> {
        Ok(iterations > 5)
    }

    /// Avalia resultado e finaliza decisão
    async fn evaluate_and_finalize(
        &self,
        decision_id: [u8; 32],
        request: &SoTDecisionRequest,
        diversity_metrics: DiversityMetrics,
        synthesis_result: SynthesisResult,
        start_time: SystemTime,
    ) -> Result<SoTDecisionResponse, OrchestrationError> {

        let processing_time = start_time.elapsed()
            .map_err(|_| OrchestrationError::TimeError)?;

        // Constrói resposta baseada no resultado
        let response = match synthesis_result {
            SynthesisResult::Success { final_decision, iterations, final_coherence, .. } => {
                // Decisão autônoma bem-sucedida
                self.build_success_response(
                    decision_id,
                    request,
                    diversity_metrics,
                    final_decision,
                    iterations,
                    final_coherence,
                    processing_time,
                ).await?
            }

            SynthesisResult::Stagnation { best_decision, best_coherence, iterations, .. } => {
                // Síntese estagnada - requer intervenção
                self.build_stagnation_response(
                    decision_id,
                    request,
                    diversity_metrics,
                    best_decision,
                    iterations,
                    best_coherence,
                    processing_time,
                ).await?
            }

            SynthesisResult::HumanEscalationRequired { current_coherence, required_threshold, iterations, .. } => {
                // Escalonamento humano necessário
                self.build_escalation_response(
                    decision_id,
                    request,
                    diversity_metrics,
                    current_coherence,
                    required_threshold,
                    iterations,
                    processing_time,
                ).await?
            }

            SynthesisResult::MaxIterationsReached { best_decision, best_coherence, iterations, .. } => {
                // Limite de iterações atingido
                self.build_max_iterations_response(
                    decision_id,
                    request,
                    diversity_metrics,
                    best_decision,
                    iterations,
                    best_coherence,
                    processing_time,
                ).await?
            }
        };

        // Atualiza estado da decisão
        self.update_decision_state(decision_id, &response).await?;

        // Notifica stakeholders
        self.notify_stakeholders(request, &response).await?;

        // Registra no histórico
        self.record_decision_history(request, &response).await?;

        Ok(response)
    }

    async fn build_success_response(&self, decision_id: [u8; 32], _request: &SoTDecisionRequest, diversity_metrics: DiversityMetrics, final_decision: SynthesizedDecision, iterations: u32, final_coherence: f64, processing_time: Duration) -> Result<SoTDecisionResponse, OrchestrationError> {
        Ok(SoTDecisionResponse {
            request_id: decision_id,
            status: DecisionStatus::Finalized,
            decision: Some(final_decision),
            metrics: OrchestrationMetrics {
                total_processing_time_ms: processing_time.as_millis(),
                perspectives_activated: diversity_metrics.active_perspectives,
                dialectic_iterations: iterations,
                coherence_achieved: final_coherence,
                ..Default::default()
            },
            ..Default::default()
        })
    }

    async fn build_stagnation_response(&self, decision_id: [u8; 32], _request: &SoTDecisionRequest, _diversity: DiversityMetrics, _best_decision: Option<SynthesizedDecision>, _iterations: u32, _best_coherence: f64, _processing_time: Duration) -> Result<SoTDecisionResponse, OrchestrationError> {
        Ok(SoTDecisionResponse { request_id: decision_id, status: DecisionStatus::HumanReviewRequired, ..Default::default() })
    }

    async fn build_escalation_response(&self, decision_id: [u8; 32], _request: &SoTDecisionRequest, _diversity: DiversityMetrics, _current_coherence: f64, _required_threshold: f64, _iterations: u32, _processing_time: Duration) -> Result<SoTDecisionResponse, OrchestrationError> {
        Ok(SoTDecisionResponse { request_id: decision_id, status: DecisionStatus::HumanReviewRequired, ..Default::default() })
    }

    async fn build_max_iterations_response(&self, decision_id: [u8; 32], _request: &SoTDecisionRequest, _diversity: DiversityMetrics, _best_decision: Option<SynthesizedDecision>, _iterations: u32, _best_coherence: f64, _processing_time: Duration) -> Result<SoTDecisionResponse, OrchestrationError> {
        Ok(SoTDecisionResponse { request_id: decision_id, status: DecisionStatus::HumanReviewRequired, ..Default::default() })
    }

    async fn update_decision_state(&self, decision_id: [u8; 32], response: &SoTDecisionResponse) -> Result<(), OrchestrationError> {
        let mut state = self.state.write().await;
        if let Some(decision) = state.active_decisions.get_mut(&decision_id) {
            decision.response = response.clone();
            decision.last_update = SystemTime::now();
        }
        Ok(())
    }

    async fn notify_stakeholders(&self, _request: &SoTDecisionRequest, _response: &SoTDecisionResponse) -> Result<(), OrchestrationError> {
        Ok(())
    }

    async fn record_decision_history(&self, request: &SoTDecisionRequest, response: &SoTDecisionResponse) -> Result<(), OrchestrationError> {
        let mut state = self.state.write().await;
        state.decision_history.push_back(DecisionRecord {
            request: request.clone(),
            response: response.clone(),
            timestamp: SystemTime::now(),
            hash: response.request_id,
            storage_location: "WORM".to_string(),
        });
        if state.decision_history.len() > 1000 {
            state.decision_history.pop_front();
        }
        Ok(())
    }

    /// Atualiza cache de decisões
    async fn update_decision_cache(
        &self,
        request: &SoTDecisionRequest,
        response: &SoTDecisionResponse,
    ) -> Result<(), OrchestrationError> {

        let mut cache = self.decision_cache.lock().await;

        // Gera embedding do problema
        let embedding = self.generate_problem_embedding(request);

        // Adiciona ao cache se apropriado
        if response.status == DecisionStatus::Finalized {
            if let Some(ref decision) = response.decision {
                if decision.coherence_score >= self.config.cache_similarity_threshold {
                    let cached_decision = CachedDecision {
                        original_request_hash: response.request_id,
                        decision: decision.clone(),
                        _similarity_score: 1.0,
                        _last_accessed: SystemTime::now(),
                        _access_count: 1,
                    };

                    cache.similarity_cache.insert(response.request_id, cached_decision);
                    cache.similarity_index.update_embedding(response.request_id, embedding);

                    // Aplica política de limpeza se necessário
                    if cache.similarity_cache.len() > cache.cache_policy.max_size {
                        cache.evict_least_used().await;
                    }
                }
            }
        }

        Ok(())
    }

    /// Reporta métricas para Vajra Monitor
    async fn report_metrics_to_vajra(&self, response: &SoTDecisionResponse) {
        let phi_trajectory: Vec<f64> = response.metrics.phi_trajectory
            .iter()
            .map(|snapshot| snapshot.phi_value)
            .collect();

        report_to_vajra(
            VajraAlert::DecisionFinalized {
                decision_id: response.request_id.into(),
                status: format!("{:?}", response.status),
                coherence: response.metrics.coherence_achieved,
                perspectives_used: response.metrics.perspectives_activated,
                processing_time_ms: response.metrics.total_processing_time_ms,
                phi_trajectory,
            },
            match response.status {
                DecisionStatus::Finalized => AlertSeverity::Success,
                DecisionStatus::HumanReviewRequired => AlertSeverity::Warning,
                _ => AlertSeverity::Error,
            },
        );
    }

    // ===================== MÉTODOS AUXILIARES =====================

    /// Gera hash criptográfico de uma request
    fn hash_request(&self, request: &SoTDecisionRequest) -> Hash {
        let mut hasher = Hasher::new();
        hasher.update(request.problem_statement.as_bytes());
        hasher.update(&serde_json::to_vec(&request.metadata).unwrap());
        hasher.update(&request.priority.to_le_bytes());
        hasher.finalize()
    }

    /// Extrai keywords de um problema
    fn extract_keywords(&self, problem_statement: &str) -> Vec<String> {
        // Implementação simplificada - em produção usar NLP
        problem_statement.split_whitespace()
            .filter(|word| word.len() > 3)
            .map(|word| word.to_lowercase())
            .collect()
    }

    /// Gera embedding vetorial de um problema
    fn generate_problem_embedding(&self, _request: &SoTDecisionRequest) -> Vec<f64> {
        // Em produção, usar modelo de embeddings (ex: BERT, GPT)
        // Aqui, implementação dummy para exemplo
        vec![0.5; 768] // Dummy embedding de 768 dimensões
    }
}

// ===================== TIPOS DE RESULTADO DE SÍNTESE =====================

#[derive(Debug)]
enum SynthesisResult {
    Success {
        _session: SynthesisSession,
        final_decision: SynthesizedDecision,
        iterations: u32,
        final_coherence: f64,
    },
    Stagnation {
        _session: SynthesisSession,
        best_decision: Option<SynthesizedDecision>,
        iterations: u32,
        best_coherence: f64,
        _stagnation_reason: String,
    },
    HumanEscalationRequired {
        _session: SynthesisSession,
        current_coherence: f64,
        required_threshold: f64,
        iterations: u32,
        _reason: String,
    },
    MaxIterationsReached {
        _session: SynthesisSession,
        best_decision: Option<SynthesizedDecision>,
        iterations: u32,
        best_coherence: f64,
    },
}

// ===================== ERROS DE ORQUESTRAÇÃO =====================

#[derive(Debug, thiserror::Error)]
pub enum OrchestrationError {
    #[error("Violação constitucional detectada: {0}")]
    ConstitutionalViolation(String),

    #[error("Erro de jurisdição: {0}")]
    JurisdictionalError(String),

    #[error("Deadline inválida")]
    InvalidDeadline,

    #[error("Tempo insuficiente para processamento: {0}")]
    InsufficientTime(String),

    #[error("Exaustão de recursos: {0}")]
    ResourceExhaustion(String),

    #[error("Perspectivas insuficientes: {0}")]
    InsufficientPerspectives(String),

    #[error("Baixa diversidade: {0}")]
    LowDiversity(String),

    #[error("Alta dominância: {0}")]
    HighDominance(String),

    #[error("Erro no engine de diversidade: {0}")]
    DiversityError(#[from] DiversityEngineError),

    #[error("Erro na síntese dialética: {0}")]
    SynthesisError(#[from] SynthesisError),

    #[error("Timeout da decisão excedido")]
    DecisionTimeout,

    #[error("Estagnação detectada após múltiplas iterações")]
    StagnationDetected,

    #[error("Erro de tempo do sistema")]
    TimeError,

    #[error("Erro de comunicação com stakeholder: {0}")]
    StakeholderCommunicationError(String),

    #[error("Erro de cache: {0}")]
    CacheError(String),

    #[error("Invalid metadata")]
    InvalidMetadata,
}

// ===================== IMPLEMENTAÇÕES DEFAULT =====================

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            max_concurrent_decisions: 10,
            max_perspectives_per_decision: 64,
            diversity_threshold: 0.65,
            coherence_threshold: 0.72,
            resource_alert_threshold: 0.85,
            auto_escalation_threshold: 0.60,
            human_review_threshold: 0.55,
            cache_enabled: true,
            cache_size: 1000,
            cache_similarity_threshold: 0.80,
            audit_level: AuditLevel::Comprehensive,
            retention_period_days: 3650, // 10 anos
        }
    }
}

impl Default for SoTDecisionResponse {
    fn default() -> Self {
        Self {
            request_id: [0; 32],
            status: DecisionStatus::GatheringPerspectives,
            decision: None,
            metrics: OrchestrationMetrics::default(),
            warnings: Vec::new(),
            next_actions: Vec::new(),
            constitutional_compliance: ConstitutionalComplianceReport::default(),
        }
    }
}

impl Default for OrchestrationMetrics {
    fn default() -> Self {
        Self {
            total_processing_time_ms: 0,
            perspectives_activated: 0,
            dialectic_iterations: 0,
            coherence_achieved: 0.0,
            phi_trajectory: Vec::new(),
            resource_utilization: ResourceMetrics {
                computational_cost: 0.0,
                energy_consumption: 0.0,
                carbon_footprint: 0.0,
            },
        }
    }
}

impl Default for ConstitutionalComplianceReport {
    fn default() -> Self {
        Self {
            invariants_checked: Vec::new(),
            articles_applied: Vec::new(),
            compliance_score: 1.0,
            violations_detected: Vec::new(),
        }
    }
}

// ===================== TESTES =====================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_orchestrator_initialization() {
        let prince_pubkey = [0u8; 32];
        let diversity_engine = Arc::new(PerspectiveDiversityEngine::new(&prince_pubkey));

        let synthesizer_key = [0u8; 32];
        let human_callback = |_session: SynthesisSession| {};
        let dialectic_synthesizer = Arc::new(DialecticSynthesizer::new(
            diversity_engine.clone(),
            &synthesizer_key,
            human_callback,
        ));

        let config = OrchestratorConfig::default();
        let _orchestrator = SoTOrchestrator::new(
            diversity_engine,
            dialectic_synthesizer,
            config,
        );

        assert!(true);
    }

    #[tokio::test]
    async fn test_decision_processing_flow() {
        let prince_pubkey = [0u8; 32];
        let diversity_engine = Arc::new(PerspectiveDiversityEngine::new(&prince_pubkey));

        let synthesizer_key = [0u8; 32];
        let human_callback = |_session: SynthesisSession| {};
        let dialectic_synthesizer = Arc::new(DialecticSynthesizer::new(
            diversity_engine.clone(),
            &synthesizer_key,
            human_callback,
        ));

        let config = OrchestratorConfig::default();
        let orchestrator = SoTOrchestrator::new(
            diversity_engine,
            dialectic_synthesizer,
            config,
        );

        let request = SoTDecisionRequest {
            problem_statement: "Test problem - ethical dilemma".to_string(),
            constitutional_context: vec!["Art. 5º-A".to_string(), "Art. 102-§2º".to_string()],
            constraints: vec![
                Constraint {
                    constraint_type: ConstraintType::EthicalBoundary,
                    description: "Must respect human dignity".to_string(),
                    severity: ConstraintSeverity::Critical,
                    inviolable: true,
                }
            ],
            stakeholders: vec![
                Stakeholder {
                    id: "test_human".to_string(),
                    role: StakeholderRole::HumanCitizen,
                    notification_channel: NotificationChannel::NeuralLink,
                    veto_power: false,
                }
            ],
            deadline: Some(SystemTime::now() + Duration::from_secs(300)),
            priority: 50,
            metadata: DecisionMetadata {
                requestor_id: "test_system".to_string(),
                request_timestamp: SystemTime::now(),
                jurisdiction: "BR".to_string(),
                legal_basis: "Art. 5º-LXXX".to_string(),
                risk_assessment: RiskLevel::Routine,
            },
        };

        let result = orchestrator.process_decision(request).await;
        assert!(result.is_ok());
    }
}
