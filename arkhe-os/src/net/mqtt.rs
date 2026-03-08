//! MQTT Adapter for ArkheNet
//! Enables ingestion of industrial and IoT sensor data into the temporal substrate.

use rumqttc::{AsyncClient, MqttOptions, QoS};
use std::time::Duration;
use tracing::{info, error};

pub struct ArkheMqttClient {
    pub client: AsyncClient,
}

impl ArkheMqttClient {
    pub async fn new(host: &str, port: u16, client_id: &str) -> Self {
        let mut mqttoptions = MqttOptions::new(client_id, host, port);
        mqttoptions.set_keep_alive(Duration::from_secs(5));

        let (client, mut eventloop) = AsyncClient::new(mqttoptions, 10);

        // Spawn event loop handler
        tokio::spawn(async move {
            loop {
                match eventloop.poll().await {
                    Ok(notification) => {
                        // info!("[MQTT] Notification: {:?}", notification);
                    }
                    Err(e) => {
                        error!("[MQTT] Error: {:?}", e);
                        tokio::time::sleep(Duration::from_secs(1)).await;
                    }
                }
            }
        });

        Self { client }
    }

    pub async fn subscribe_to_industrial_signals(&self) -> anyhow::Result<()> {
        self.client.subscribe("arkhe/industrial/+", QoS::AtMostOnce).await?;
        info!("[MQTT] Subscribed to arkhe/industrial/+");
        Ok(())
    }

    pub async fn publish_temporal_status(&self, payload: &str) -> anyhow::Result<()> {
        self.client.publish("arkhe/status/temporal", QoS::AtLeastOnce, false, payload).await?;
        Ok(())
    }
}
