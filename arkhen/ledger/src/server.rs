use tonic::{Request, Response, Status};
use crate::core::{LedgerCore, Handover};
use std::sync::Arc;
use tokio::sync::RwLock;

pub mod ledger_proto {
    tonic::include_proto!("ledger");
}

use ledger_proto::{
    ledger_server::Ledger, AppendRequest, AppendResponse, QueryRequest, QueryResponse,
    StatusResponse, Empty,
};

pub struct LedgerService {
    pub core: Arc<RwLock<LedgerCore>>,
}

#[tonic::async_trait]
impl Ledger for LedgerService {
    async fn append(&self, req: Request<AppendRequest>) -> Result<Response<AppendResponse>, Status> {
        let proto_h = req.into_inner().handover.ok_or_else(|| Status::invalid_argument("missing handover"))?;

        let handover = Handover {
            id: proto_h.id,
            type_: proto_h.r#type,
            emitter_id: proto_h.emitter_id,
            receiver_id: proto_h.receiver_id,
            entropy_cost: proto_h.entropy_cost,
            half_life: proto_h.half_life,
            payload: proto_h.payload,
            timestamp_physical: proto_h.timestamp_physical,
            timestamp_logical: proto_h.timestamp_logical,
            signature: proto_h.signature,
        };

        let mut core = self.core.write().await;
        let (hash, index) = core.append(handover).map_err(|e| Status::internal(e))?;

        Ok(Response::new(AppendResponse { hash, index }))
    }

    async fn query(&self, req: Request<QueryRequest>) -> Result<Response<QueryResponse>, Status> {
        let params = req.into_inner();
        let core = self.core.read().await;
        let start = params.start_index.unwrap_or(0) as usize;
        let end = params.end_index.unwrap_or(core.len()) as usize;
        let blocks = core.range(start, end);

        let handovers = blocks.iter().map(|b| {
            ledger_proto::Handover {
                id: b.handover.id.clone(),
                r#type: b.handover.type_,
                emitter_id: b.handover.emitter_id,
                receiver_id: b.handover.receiver_id,
                entropy_cost: b.handover.entropy_cost,
                half_life: b.handover.half_life,
                payload: b.handover.payload.clone(),
                timestamp_physical: b.handover.timestamp_physical,
                timestamp_logical: b.handover.timestamp_logical,
                signature: b.handover.signature.clone(),
            }
        }).collect();

        Ok(Response::new(QueryResponse { handovers }))
    }

    async fn get_status(&self, _: Request<Empty>) -> Result<Response<StatusResponse>, Status> {
        let core = self.core.read().await;
        Ok(Response::new(StatusResponse {
            length: core.len(),
            last_hash: core.last_hash().unwrap_or_default(),
            total_entropy: core.total_entropy(),
            uptime_ns: 0,
        }))
    }
}
