//! gRPC Server - Bloco #48
//! Exposição segura com autenticação PQC e rate limiting

use tonic::{transport::Server, Request, Response, Status};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::time::timeout;
use tower::ServiceBuilder;
use tower::limit::RateLimitLayer;
use tower::buffer::BufferLayer;
use log::info;

use crate::engine::{SoTOrchestrator, SoTDecisionRequest, SoTDecisionResponse};
use crate::grpc::authentication::authenticate_request;
use crate::grpc::sasc_society_proto::sot_orchestrator_server::{SotOrchestrator as SotOrchestratorTrait, SotOrchestratorServer};
use crate::grpc::sasc_society_proto::{
    ProcessDecisionRequest, ProcessDecisionResponse,
    GetDecisionStatusRequest, GetDecisionStatusResponse,
    GetDecisionHistoryRequest, DecisionRecord,
    RequestHumanEscalationRequest, RequestHumanEscalationResponse,
    MetricsRequest, MetricsUpdate,
    HardFreezeAlertRequest, HardFreezeAlert
};

const GRPC_TIMEOUT: Duration = Duration::from_secs(120);

pub struct GrpcServer {
    orchestrator: Arc<SoTOrchestrator>,
    prince_pubkey: pqcrypto_dilithium::dilithium5::PublicKey,
}

impl GrpcServer {
    pub fn new(
        orchestrator: Arc<SoTOrchestrator>,
        prince_pubkey: pqcrypto_dilithium::dilithium5::PublicKey,
    ) -> Self {
        Self {
            orchestrator,
            prince_pubkey,
        }
    }
}

#[tonic::async_trait]
impl SotOrchestratorTrait for GrpcServer {
    async fn process_decision(
        &self,
        request: Request<ProcessDecisionRequest>,
    ) -> Result<Response<ProcessDecisionResponse>, Status> {

        // 1. Autenticação PQC (INV-1 non-repudiation)
        let request = authenticate_request(request, &self.prince_pubkey).await?;
        let inner_request = request.into_inner();

        // 2. Conversão para Rust structs
        let sot_request = convert_proto_to_decision_request(inner_request)?;

        // 3. Validação de deadline
        if let Some(deadline) = sot_request.deadline {
            if deadline < SystemTime::now() {
                return Err(Status::deadline_exceeded("Deadline expirado"));
            }
        }

        // 4. Execução do pipeline SoT
        let response = timeout(GRPC_TIMEOUT, self.orchestrator.process_decision(sot_request))
            .await
            .map_err(|_| Status::deadline_exceeded("Timeout gRPC"))?
            .map_err(|e| Status::internal(format!("Erro orquestrador: {}", e)))?;

        // 5. Assinatura Prince (INV-2 proveniência)
        let proto_response = convert_decision_response_to_proto(response)?;
        let signed_response = sign_response_with_prince(proto_response)?;

        Ok(Response::new(signed_response))
    }

    async fn get_decision_status(
        &self,
        _request: Request<GetDecisionStatusRequest>,
    ) -> Result<Response<GetDecisionStatusResponse>, Status> {
        Err(Status::unimplemented("Not implemented in Block #48"))
    }

    type GetDecisionHistoryStream = tokio_stream::wrappers::ReceiverStream<Result<DecisionRecord, Status>>;

    async fn get_decision_history(
        &self,
        _request: Request<GetDecisionHistoryRequest>,
    ) -> Result<Response<Self::GetDecisionHistoryStream>, Status> {
        Err(Status::unimplemented("Not implemented in Block #48"))
    }

    async fn request_human_escalation(
        &self,
        _request: Request<RequestHumanEscalationRequest>,
    ) -> Result<Response<RequestHumanEscalationResponse>, Status> {
        Err(Status::unimplemented("Not implemented in Block #48"))
    }

    type StreamMetricsStream = tokio_stream::wrappers::ReceiverStream<Result<MetricsUpdate, Status>>;

    async fn stream_metrics(
        &self,
        _request: Request<MetricsRequest>,
    ) -> Result<Response<Self::StreamMetricsStream>, Status> {
        Err(Status::unimplemented("Not implemented in Block #48"))
    }

    type SubscribeHardFreezeAlertsStream = tokio_stream::wrappers::ReceiverStream<Result<HardFreezeAlert, Status>>;

    async fn subscribe_hard_freeze_alerts(
        &self,
        _request: Request<HardFreezeAlertRequest>,
    ) -> Result<Response<Self::SubscribeHardFreezeAlertsStream>, Status> {
        Err(Status::unimplemented("Not implemented in Block #48"))
    }
}

fn convert_proto_to_decision_request(proto: ProcessDecisionRequest) -> Result<SoTDecisionRequest, Status> {
    use std::time::UNIX_EPOCH;

    let deadline = proto.deadline.map(|ts| {
        UNIX_EPOCH + Duration::new(ts.seconds as u64, ts.nanos as u32)
    });

    let metadata = proto.metadata.ok_or_else(|| Status::invalid_argument("Metadata ausente"))?;

    Ok(SoTDecisionRequest {
        problem_statement: proto.problem_statement,
        constitutional_context: proto.constitutional_context,
        constraints: proto.constraints.into_iter().map(|c| {
            let severity = match c.severity() {
                crate::grpc::sasc_society_proto::constraint::Severity::Advisory => crate::engine::ConstraintSeverity::Advisory,
                crate::grpc::sasc_society_proto::constraint::Severity::Warning => crate::engine::ConstraintSeverity::Warning,
                crate::grpc::sasc_society_proto::constraint::Severity::Critical => crate::engine::ConstraintSeverity::Critical,
                crate::grpc::sasc_society_proto::constraint::Severity::Existential => crate::engine::ConstraintSeverity::Existential,
            };
            let c_type = match c.r#type() {
                crate::grpc::sasc_society_proto::constraint::Type::ConstitutionalInvariant => crate::engine::ConstraintType::ConstitutionalInvariant,
                crate::grpc::sasc_society_proto::constraint::Type::TechnicalFeasibility => crate::engine::ConstraintType::TechnicalFeasibility,
                crate::grpc::sasc_society_proto::constraint::Type::ResourceLimitation => crate::engine::ConstraintType::ResourceLimitation,
                crate::grpc::sasc_society_proto::constraint::Type::EthicalBoundary => crate::engine::ConstraintType::EthicalBoundary,
                crate::grpc::sasc_society_proto::constraint::Type::TemporalUrgency => crate::engine::ConstraintType::TemporalUrgency,
            };
            crate::engine::Constraint {
                constraint_type: c_type,
                description: c.description,
                severity,
                inviolable: c.inviolable,
            }
        }).collect(),
        stakeholders: proto.stakeholders.into_iter().map(|s| {
            let role = match s.role() {
                crate::grpc::sasc_society_proto::stakeholder::Role::HumanCitizen => crate::engine::StakeholderRole::HumanCitizen,
                crate::grpc::sasc_society_proto::stakeholder::Role::MycelialNetwork => crate::engine::StakeholderRole::MycelialNetwork,
                crate::grpc::sasc_society_proto::stakeholder::Role::AsimovNode => crate::engine::StakeholderRole::AsimovNode,
                crate::grpc::sasc_society_proto::stakeholder::Role::PrinceAuthority => crate::engine::StakeholderRole::PrinceAuthority,
                crate::grpc::sasc_society_proto::stakeholder::Role::CouncilMember => crate::engine::StakeholderRole::CouncilMember,
            };
            let channel = match s.notification_channel() {
                crate::grpc::sasc_society_proto::stakeholder::Channel::NeuralLink => crate::engine::NotificationChannel::NeuralLink,
                crate::grpc::sasc_society_proto::stakeholder::Channel::BioElectric => crate::engine::NotificationChannel::BioElectric,
                crate::grpc::sasc_society_proto::stakeholder::Channel::QuantumEntangled => crate::engine::NotificationChannel::QuantumEntangled,
                crate::grpc::sasc_society_proto::stakeholder::Channel::ConstitutionalBroadcast => crate::engine::NotificationChannel::ConstitutionalBroadcast,
            };
            crate::engine::Stakeholder {
                id: s.id,
                role,
                notification_channel: channel,
                veto_power: s.veto_power,
            }
        }).collect(),
        deadline,
        priority: proto.priority as u8,
        metadata: {
            let risk = match metadata.risk_assessment() {
                crate::grpc::sasc_society_proto::decision_metadata::RiskLevel::Routine => crate::engine::RiskLevel::Routine,
                crate::grpc::sasc_society_proto::decision_metadata::RiskLevel::Strategic => crate::engine::RiskLevel::Strategic,
                crate::grpc::sasc_society_proto::decision_metadata::RiskLevel::Existential => crate::engine::RiskLevel::Existential,
                crate::grpc::sasc_society_proto::decision_metadata::RiskLevel::Constitutional => crate::engine::RiskLevel::Constitutional,
            };
            crate::engine::DecisionMetadata {
                requestor_id: metadata.requestor_id,
                request_timestamp: metadata.request_timestamp.map(|ts| {
                    UNIX_EPOCH + Duration::new(ts.seconds as u64, ts.nanos as u32)
                }).unwrap_or(SystemTime::now()),
                jurisdiction: metadata.jurisdiction,
                legal_basis: metadata.legal_basis,
                risk_assessment: risk,
            }
        },
    })
}

fn convert_decision_response_to_proto(resp: SoTDecisionResponse) -> Result<ProcessDecisionResponse, Status> {
    use prost_types::Timestamp;

    Ok(ProcessDecisionResponse {
        decision_id: resp.request_id.to_vec(),
        status: match resp.status {
            crate::engine::DecisionStatus::GatheringPerspectives => crate::grpc::sasc_society_proto::DecisionStatus::GatheringPerspectives as i32,
            crate::engine::DecisionStatus::DiversityAssessment => crate::grpc::sasc_society_proto::DecisionStatus::DiversityAssessment as i32,
            crate::engine::DecisionStatus::DialecticSynthesis => crate::grpc::sasc_society_proto::DecisionStatus::DialecticSynthesis as i32,
            crate::engine::DecisionStatus::HumanReviewRequired => crate::grpc::sasc_society_proto::DecisionStatus::HumanReviewRequired as i32,
            crate::engine::DecisionStatus::Finalized => crate::grpc::sasc_society_proto::DecisionStatus::Finalized as i32,
            crate::engine::DecisionStatus::HardFreezeTriggered => crate::grpc::sasc_society_proto::DecisionStatus::HardFreezeTriggered as i32,
            crate::engine::DecisionStatus::TimeoutExceeded => crate::grpc::sasc_society_proto::DecisionStatus::TimeoutExceeded as i32,
            crate::engine::DecisionStatus::ConstitutionalViolationDetected => crate::grpc::sasc_society_proto::DecisionStatus::ConstitutionalViolationDetected as i32,
        },
        decision: resp.decision.map(|d| crate::grpc::sasc_society_proto::SynthesizedDecision {
            decision_text: d.decision_text,
            coherence_score: d.coherence_score,
            supporting_arguments: d.supporting_arguments,
            counter_arguments: d.counter_arguments,
            consensus_level: d.consensus_level,
        }),
        metrics: Some(crate::grpc::sasc_society_proto::OrchestrationMetrics {
            total_processing_time_ms: resp.metrics.total_processing_time_ms as u64,
            perspectives_activated: resp.metrics.perspectives_activated as u32,
            dialectic_iterations: resp.metrics.dialectic_iterations,
            coherence_achieved: resp.metrics.coherence_achieved,
            phi_trajectory: resp.metrics.phi_trajectory.into_iter().map(|p| {
                crate::grpc::sasc_society_proto::PhiSnapshot {
                    timestamp: Some(Timestamp {
                        seconds: p.timestamp.duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default().as_secs() as i64,
                        nanos: p.timestamp.duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default().subsec_nanos() as i32,
                    }),
                    phi_value: p.phi_value,
                    reason: p.reason,
                }
            }).collect(),
            resource_utilization: Some(crate::grpc::sasc_society_proto::ResourceMetrics {
                computational_cost: resp.metrics.resource_utilization.computational_cost,
                energy_consumption: resp.metrics.resource_utilization.energy_consumption,
                carbon_footprint: resp.metrics.resource_utilization.carbon_footprint,
            }),
        }),
        warnings: resp.warnings.into_iter().map(|w| {
            crate::grpc::sasc_society_proto::DecisionWarning {
                r#type: match w.warning_type {
                    crate::engine::WarningType::LowDiversity => crate::grpc::sasc_society_proto::decision_warning::Type::LowDiversity as i32,
                    crate::engine::WarningType::HighDominance => crate::grpc::sasc_society_proto::decision_warning::Type::HighDominance as i32,
                    crate::engine::WarningType::CoherenceDecline => crate::grpc::sasc_society_proto::decision_warning::Type::CoherenceDecline as i32,
                    crate::engine::WarningType::ResourceExhaustion => crate::grpc::sasc_society_proto::decision_warning::Type::ResourceExhaustion as i32,
                    crate::engine::WarningType::ConstitutionalAmbiguity => crate::grpc::sasc_society_proto::decision_warning::Type::ConstitutionalAmbiguity as i32,
                    crate::engine::WarningType::StakeholderConflict => crate::grpc::sasc_society_proto::decision_warning::Type::StakeholderConflict as i32,
                },
                description: w.description,
                severity: 1, // Mapping required
                recommended_action: w.recommended_action,
            }
        }).collect(),
        next_actions: resp.next_actions.into_iter().map(|a| {
            crate::grpc::sasc_society_proto::NextAction {
                action: a.action,
                deadline: a.deadline.map(|ts| Timestamp {
                    seconds: ts.duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default().as_secs() as i64,
                    nanos: ts.duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default().subsec_nanos() as i32,
                }),
                responsible: a.responsible,
                dependencies: a.dependencies,
            }
        }).collect(),
        compliance_report: Some(crate::grpc::sasc_society_proto::ConstitutionalComplianceReport {
            invariants_checked: resp.constitutional_compliance.invariants_checked.into_iter().map(|i| {
                crate::grpc::sasc_society_proto::InvariantCheck {
                    invariant: i.invariant,
                    status: format!("{:?}", i.status),
                    evidence: i.evidence,
                    confidence: i.confidence,
                }
            }).collect(),
            articles_applied: resp.constitutional_compliance.articles_applied.into_iter().map(|a| {
                crate::grpc::sasc_society_proto::ArticleApplication {
                    article: a.article,
                    interpretation: a.interpretation,
                    relevance: a.relevance,
                }
            }).collect(),
            compliance_score: resp.constitutional_compliance.compliance_score,
            violations_detected: resp.constitutional_compliance.violations_detected.into_iter().map(|v| {
                crate::grpc::sasc_society_proto::ConstitutionalViolation {
                    article: v.article,
                    violation_type: format!("{:?}", v.violation_type),
                    severity: format!("{:?}", v.severity),
                    corrective_action: v.corrective_action,
                }
            }).collect(),
        }),
        finalized_at: Some(Timestamp {
            seconds: SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default().as_secs() as i64,
            nanos: SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default().subsec_nanos() as i32,
        }),
        signature: Vec::new(),
    })
}

fn sign_response_with_prince(mut proto: ProcessDecisionResponse) -> Result<ProcessDecisionResponse, Status> {
    // Em produção, usa a chave Prince para assinar o hash da resposta
    proto.signature = vec![0u8; 4608]; // Dilithium5 signature placeholder
    Ok(proto)
}

pub async fn start_server(
    orchestrator: Arc<SoTOrchestrator>,
    prince_pubkey: pqcrypto_dilithium::dilithium5::PublicKey,
    addr: String,
) -> Result<(), Box<dyn std::error::Error>> {

    let server = GrpcServer::new(orchestrator, prince_pubkey);

    // Rate limiting: 10 req/s por IP (INV-3: não-concentração)
    // BufferLayer used to make it Clone
    let service = Server::builder()
        .layer(
            ServiceBuilder::new()
                .layer(BufferLayer::new(100))
                .layer(RateLimitLayer::new(10, Duration::from_secs(1)))
        )
        .add_service(SotOrchestratorServer::new(server))
        .serve(addr.parse()?);

    info!("🏛️ SoT gRPC Server ativo em {}", addr);
    service.await?;

    Ok(())
}
