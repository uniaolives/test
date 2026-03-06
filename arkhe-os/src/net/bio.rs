use tokio::net::UdpSocket;
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::Mutex;
use crate::kernel::scheduler::CoherenceScheduler;

#[derive(Deserialize)]
struct AmbientData {
    variance: f64,
}

pub struct BioAntenna {
    port: u16,
}

impl BioAntenna {
    pub fn new(port: u16) -> Self {
        Self { port }
    }

    pub async fn run(&self, scheduler: Arc<Mutex<CoherenceScheduler>>) {
        let socket = UdpSocket::bind(format!("127.0.0.1:{}", self.port)).await.expect("Falha ao bindar UDP BioAntenna");
        println!("[BIO] Antena Biocibernética ativa na porta UDP {}", self.port);

        let mut buf = [0; 1024];
        let mut moving_average_variance = 0.0;
        let alpha = 0.1;

        loop {
            if let Ok((len, _)) = socket.recv_from(&mut buf).await {
                let msg = String::from_utf8_lossy(&buf[..len]);
                if let Ok(data) = serde_json::from_str::<AmbientData>(&msg) {
                    if moving_average_variance == 0.0 {
                        moving_average_variance = data.variance;
                    } else {
                        moving_average_variance = (alpha * data.variance) + ((1.0 - alpha) * moving_average_variance);
                    }

                    if data.variance > moving_average_variance * 1.5 {
                        println!("\n[BIO-ANOMALIA] Pico detectado! Var: {:.2} > Média: {:.2}", data.variance, moving_average_variance);
                        let mut sched = scheduler.lock().await;
                        sched.inject_coherence(0.1);
                    }
                }
            }
        }
    }
}
