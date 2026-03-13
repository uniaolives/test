// arkhe-os/src/network/manager.rs

use super::transport::{TzinorTransport, TzinorConfig};
use super::protocol::{Http4Method, OrbRequest, OrbResponse, TemporalHeaders};
use super::router::TemporalRouter;
use chrono::Utc;

pub struct OrbNetworkManager {
    transport: TzinorTransport,
    router: TemporalRouter,
}

impl OrbNetworkManager {
    pub fn new() -> Self {
        Self {
            transport: TzinorTransport::new(TzinorConfig::default()),
            router: TemporalRouter::new(),
        }
    }

    pub async fn process_request(&mut self, req: OrbRequest) -> OrbResponse {
        if req.headers.x_lambda_2 < 0.618 {
            return OrbResponse::insufficient_coherence();
        }

        let shards = self.transport.shard_data(&req.payload);

        match self.router.route(shards[0].clone(), &req.headers.x_temporal_target) {
            Ok(_) => OrbResponse::accepted(),
            Err(e) => OrbResponse {
                status_code: 451,
                status_message: format!("Paradox or Routing Error: {}", e),
                data: None,
            }
        }
    }

    pub async fn emit_orb(&mut self, data: Vec<u8>, target: &str, coherence: f64) -> OrbResponse {
        let req = OrbRequest {
            method: Http4Method::EMIT,
            headers: TemporalHeaders {
                x_temporal_origin: Utc::now().to_rfc3339(),
                x_temporal_target: target.to_string(),
                x_lambda_2: coherence,
                x_oam_index: 5,
            },
            payload: data,
        };
        self.process_request(req).await
    }
}
