use serde::{Deserialize, Serialize};

/// Comandos do protocolo Teknet P2P
#[derive(Debug, Serialize, Deserialize)]
pub enum TeknetMessage {
    /// Handshake inicial: "Olá, eu sou o nó X e estou no bloco Y"
    Hello {
        peer_id: String,
        last_handover_id: u64
    },

    /// Solicitação de sincronização: "Me envie tudo do bloco X ao Y"
    SyncRequest {
        from_id: u64,
        to_id: u64
    },

    /// Resposta com dados: "Aqui estão os blocos solicitados"
    SyncResponse {
        handovers: Vec<HandoverData>
    },

    /// Broadcast de novo evento: "Acabei de gravar isso"
    NewHandover {
        handover: HandoverData
    },
}

/// Estrutura de dados para transmissão (serializada)
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct HandoverData {
    pub id: u64,
    pub timestamp: i64,
    pub description: String,
    pub phi_q_after: f64,
}
