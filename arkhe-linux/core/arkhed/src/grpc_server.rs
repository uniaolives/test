use tonic::{Request, Response, Status};
use arkhe::arkhe_service_server::ArkheService;
use arkhe::{HandoverRequest, StatusRequest, StatusResponse, OntologyRequest, ArkheResponse};
use std::sync::Arc;
use crate::ArkheSystem;

pub mod arkhe {
    tonic::include_proto!("arkhe");
}

pub struct ArkheServiceImpl {
    pub system: Arc<ArkheSystem>,
}

#[tonic::async_trait]
impl ArkheService for ArkheServiceImpl {
    async fn send_handover(&self, request: Request<HandoverRequest>) -> Result<Response<ArkheResponse>, Status> {
        let req = request.into_inner();
        tracing::info!("gRPC: Handover received ({} bytes)", req.data.len());

        // Simulação de processamento
        Ok(Response::new(ArkheResponse {
            success: true,
            message: "Handover queued".into(),
        }))
    }

    async fn get_status(&self, _request: Request<StatusRequest>) -> Result<Response<StatusResponse>, Status> {
        let phi = *self.system.phi.read().await;
        Ok(Response::new(StatusResponse {
            phi,
            entropy: 0.618,
        }))
    }

    async fn update_ontology(&self, request: Request<OntologyRequest>) -> Result<Response<ArkheResponse>, Status> {
        let req = request.into_inner();
        tracing::info!("gRPC: Ontology Update - {} ({})", req.object_id, req.object_type);

        Ok(Response::new(ArkheResponse {
            success: true,
            message: "Ontology synchronized".into(),
        }))
    }
}
