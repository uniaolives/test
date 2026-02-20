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
        let expected_next = self.last_processed + 1_000_000;
        ts == expected_next
    }

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

pub struct BufferedAnyonicInterceptor {
    pub inner: AnyonicInterceptor,
    pub buffer: BraidingBuffer,
}

impl Interceptor for BufferedAnyonicInterceptor {
    fn call(&mut self, request: Request<()>) -> Result<Request<()>, Status> {
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

        let packet = AnyonicPacket {
            timestamp_ns,
            alpha: (alpha_num, alpha_den),
            phase: Complex64::new(phase_re, phase_im),
            payload: vec![],
            received_at: Instant::now(),
        };

        self.buffer.insert_sync(packet)?;

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

        let out1 = rx.recv().await.unwrap();
        assert_eq!(out1.timestamp_ns, 1_000_000);

        let out2 = rx.recv().await.unwrap();
        assert_eq!(out2.timestamp_ns, 2_000_000);
    }
}
