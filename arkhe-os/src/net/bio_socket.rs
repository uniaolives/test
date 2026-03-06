use warp::Filter;
use futures_util::StreamExt;
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::Mutex;
use crate::kernel::scheduler::CoherenceScheduler;

#[derive(Deserialize, Debug)]
pub struct BioTelemetry {
    pub accel_variance: f64,
    #[allow(dead_code)]
    pub timestamp: u64,
}

pub async fn start_bio_server(scheduler: Arc<Mutex<CoherenceScheduler>>) {
    println!("[BIO] 📡 Ponte Biocibernética (WebSocket) iniciada na porta 3030...");

    let bio_route = warp::path("anchor")
        .and(warp::ws())
        .map(move |ws: warp::ws::Ws| {
            let sched = scheduler.clone();
            ws.on_upgrade(move |mut websocket| async move {
                println!("[BIO] 📱 Smartphone Conectado (Âncora Ativa)!");

                while let Some(result) = websocket.next().await {
                    match result {
                        Ok(msg) => {
                            if let Ok(text) = msg.to_str() {
                                if let Ok(data) = serde_json::from_str::<BioTelemetry>(text) {
                                    if data.accel_variance > 2.0 {
                                        println!("[BIO-ANOMALIA] Turbulência biológica detectada: {:.3}", data.accel_variance);
                                        let mut k = sched.lock().await;
                                        k.inject_coherence(data.accel_variance * 0.1);
                                    }
                                }
                            }
                        }
                        Err(e) => eprintln!("[BIO] Erro na conexão: {}", e),
                    }
                }
                println!("[BIO] 📴 Smartphone Desconectado.");
            })
        });

    warp::serve(bio_route).run(([0, 0, 0, 0], 3030)).await;
}
