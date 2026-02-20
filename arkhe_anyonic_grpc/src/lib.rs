//! arkhe_anyonic_grpc::buffer
//! Buffer de ordenação temporal e braiding para pacotes gRPC anyônicos

use std::collections::BTreeMap;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tonic::{Request, Status};
use num_complex::Complex64;
use tonic::service::Interceptor;

/// Pacote anyônico recebido, aguardando ordenação
#[derive(Debug, Clone)]
pub struct AnyonicPacket {
    pub timestamp_ns: u128,
    pub alpha: (u64, u64),  // (num, den)
    pub phase: Complex64,
    pub payload: Vec<u8>,
    pub received_at: Instant,
}

/// Buffer de braiding temporal com ordenação e timeout
pub struct BraidingBuffer {
    /// Pacotes ordenados por timestamp
    packets: BTreeMap<u128, AnyonicPacket>,
    /// Tamanho máximo antes de flush forçado
    max_size: usize,
    /// Timeout para espera de pacotes faltantes
    timeout: Duration,
    /// Canal de saída para pacotes processados
    output_tx: mpsc::Sender<AnyonicPacket>,
    /// Último timestamp processado (para detecção de gap)
    pub last_processed: u128,
    last_processed: u128,
}

impl BraidingBuffer {
    pub fn new(max_size: usize, timeout_ms: u64, output_tx: mpsc::Sender<AnyonicPacket>) -> Self {
        Self {
            packets: BTreeMap::new(),
            max_size,
            timeout: Duration::from_millis(timeout_ms),
            output_tx,
            last_processed: 0,
        }
    }

    /// Insere pacote no buffer
    pub fn insert_sync(&mut self, packet: AnyonicPacket) -> Result<(), Status> {
        let ts = packet.timestamp_ns;

        if ts < self.last_processed && self.last_processed - ts > 1_000_000_000 {
    /// Insere pacote no buffer, retorna pacotes prontos para processamento
    pub async fn insert(&mut self, packet: AnyonicPacket) -> Result<(), Status> {
        let ts = packet.timestamp_ns;

        // Verificar se timestamp é válido (não muito no passado)
        if ts < self.last_processed && self.last_processed - ts > 1_000_000_000 {
            // Mais de 1 segundo no passado: descartar ou processar com α neutro
            return Err(Status::out_of_range("Packet too old for topological ordering"));
        }

        self.packets.insert(ts, packet);
        self.check_flush_sync();
        Ok(())
    }

    /// Versão async para inserção
    pub async fn insert(&mut self, packet: AnyonicPacket) -> Result<(), Status> {
        let ts = packet.timestamp_ns;
        if ts < self.last_processed && self.last_processed - ts > 1_000_000_000 {
            return Err(Status::out_of_range("Packet too old for topological ordering"));
        }
        self.packets.insert(ts, packet);
        self.check_flush().await;
        Ok(())
    }

    fn check_flush_sync(&mut self) {
        let now = Instant::now();
        let ready: Vec<u128> = self.packets.iter()
            .filter(|(ts, pkt)| {
                self.packets.len() >= self.max_size ||
                now.duration_since(pkt.received_at) > self.timeout ||
                self.is_next_in_sequence(**ts)
            })
            .map(|(ts, _)| *ts)
            .collect();

        for ts in ready {
            if let Some(pkt) = self.packets.remove(&ts) {
                self.last_processed = ts;
                let _ = self.output_tx.try_send(pkt);
            }
        }
    }

    async fn check_flush(&mut self) {
        let now = Instant::now();
        let ready: Vec<u128> = self.packets.iter()
            .filter(|(ts, pkt)| {

        // Verificar se precisa de flush (tamanho ou timeout)
        self.check_flush().await;

        Ok(())
    }

    /// Verifica condições de flush e processa pacotes prontos
    async fn check_flush(&mut self) {
        let now = Instant::now();

        // Coletar pacotes que podem ser processados
        let ready: Vec<u128> = self.packets.iter()
            .filter(|(ts, pkt)| {
                // Pacote está pronto se:
                // 1. É o mais antigo e buffer está cheio, OU
                // 2. Passou do timeout, OU
                // 3. Próximo pacote na sequência está presente (gap = 1)
                self.packets.len() >= self.max_size ||
                now.duration_since(pkt.received_at) > self.timeout ||
                self.is_next_in_sequence(**ts)
            })
            .map(|(ts, _)| *ts)
            .collect();

        for ts in ready {
            if let Some(pkt) = self.packets.remove(&ts) {
                self.last_processed = ts;
                let _ = self.output_tx.send(pkt).await;
            }
        }
    }

    fn is_next_in_sequence(&self, ts: u128) -> bool {
    /// Verifica se este timestamp é o próximo na sequência esperada
    fn is_next_in_sequence(&self, ts: u128) -> bool {
        // Simplificação: assumimos granularidade de 1ms (1_000_000 ns)
        let expected_next = self.last_processed + 1_000_000;
        ts == expected_next
    }

    /// Força processamento de todos os pacotes restantes (shutdown)
    pub async fn drain(&mut self) {
        let ts_list: Vec<u128> = self.packets.keys().cloned().collect();
        for ts in ts_list {
            if let Some(pkt) = self.packets.remove(&ts) {
                self.last_processed = ts;
                let _ = self.output_tx.send(pkt).await;
            }
        }
    }
}

pub struct AnyonicInterceptor;

/// Placeholder for AnyonicInterceptor
pub struct AnyonicInterceptor;

/// Interceptor gRPC que integra o BraidingBuffer
pub struct BufferedAnyonicInterceptor {
    pub inner: AnyonicInterceptor,
    pub buffer: BraidingBuffer,
}

impl Interceptor for BufferedAnyonicInterceptor {
    fn call(&mut self, request: Request<()>) -> Result<Request<()>, Status> {
// Implementação básica do interceptor para fins de demonstração
impl BufferedAnyonicInterceptor {
    pub async fn process_request(&mut self, request: Request<()>) -> Result<Request<()>, Status> {
        // Extrair metadados anyônicos do request
        let metadata = request.metadata();
        let timestamp_ns = metadata.get("x-arkhe-timestamp-ns")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u128>().ok())
            .ok_or_else(|| Status::invalid_argument("Missing timestamp"))?;

        let alpha_num = metadata.get("x-arkhe-alpha-num")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
            .ok_or_else(|| Status::invalid_argument("Missing alpha numerator"))?;

        let alpha_den = metadata.get("x-arkhe-alpha-den")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
            .ok_or_else(|| Status::invalid_argument("Missing alpha denominator"))?;

        let phase_re = metadata.get("x-arkhe-phase-re")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(1.0);

        let phase_im = metadata.get("x-arkhe-phase-im")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);

        // Criar pacote anyônico
        let packet = AnyonicPacket {
            timestamp_ns,
            alpha: (alpha_num, alpha_den),
            phase: Complex64::new(phase_re, phase_im),
            payload: vec![],
            received_at: Instant::now(),
        };

        self.buffer.insert_sync(packet)?;

            payload: vec![], // Payload real seria extraído do request
            received_at: Instant::now(),
        };

        // Inserir no buffer (pode retornar erro se muito atrasado)
        self.buffer.insert(packet).await?;

        // O request continua, mas marcado como "buffered"
        Ok(request)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    #[tokio::test]
    async fn test_braiding_buffer_ordering() {
        let (tx, mut rx) = mpsc::channel(10);
        let mut buffer = BraidingBuffer::new(5, 100, tx);

        // Inserir pacotes fora de ordem
        let pkt1 = AnyonicPacket {
            timestamp_ns: 2_000_000,
            alpha: (1, 3),
            phase: Complex64::new(1.0, 0.0),
            payload: vec![2],
            received_at: Instant::now(),
        };
        let pkt2 = AnyonicPacket {
            timestamp_ns: 1_000_000,
            alpha: (1, 3),
            phase: Complex64::new(1.0, 0.0),
            payload: vec![1],
            received_at: Instant::now(),
        };

        buffer.insert(pkt1).await.unwrap();
        buffer.insert(pkt2).await.unwrap();

        // Como o gap é pequeno e is_next_in_sequence usa granularidade de 1ms,
        // pkt2 deve disparar o envio de pkt2 seguido de pkt1 se estiverem sequenciais.

        let out1 = rx.recv().await.unwrap();
        assert_eq!(out1.timestamp_ns, 1_000_000);

        let out2 = rx.recv().await.unwrap();
        assert_eq!(out2.timestamp_ns, 2_000_000);
    }
}
