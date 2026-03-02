use rumqttc::{AsyncClient, MqttOptions, QoS};
use serde_json::Value;
use tokio::sync::mpsc;
use std::collections::HashMap;

pub mod goggles;

#[derive(Debug, Clone)]
pub struct SensorEvent {
    pub sensor_id: String,
    pub value: f64,
    pub timestamp: std::time::Instant,
}

pub struct Actuator {
    pub id: String,
    pub topic: String,
}

pub struct DePinGateway {
    pub mqtt_client: AsyncClient,
    pub event_tx: mpsc::UnboundedSender<SensorEvent>,
    pub actuators: HashMap<String, Actuator>,
}

impl DePinGateway {
    pub async fn new(broker: &str, port: u16, client_id: &str) -> anyhow::Result<(Self, mpsc::UnboundedReceiver<SensorEvent>)> {
        let mut mqttoptions = MqttOptions::new(client_id, broker, port);
        mqttoptions.set_keep_alive(std::time::Duration::from_secs(5));
        let (client, mut eventloop) = AsyncClient::new(mqttoptions, 10);

        let (event_tx, event_rx) = mpsc::unbounded_channel();

        let tx = event_tx.clone();
        tokio::spawn(async move {
            while let Ok(notification) = eventloop.poll().await {
                if let rumqttc::Event::Incoming(rumqttc::Packet::Publish(publish)) = notification {
                    let topic = publish.topic;
                    if let Ok(value) = serde_json::from_slice::<Value>(&publish.payload) {
                        if let Some(num) = value.get("value").and_then(|v| v.as_f64()) {
                            let _ = tx.send(SensorEvent {
                                sensor_id: topic,
                                value: num,
                                timestamp: std::time::Instant::now(),
                            });
                        }
                    }
                }
            }
        });

        let gateway = DePinGateway {
            mqtt_client: client,
            event_tx,
            actuators: HashMap::new(),
        };
        Ok((gateway, event_rx))
    }

    pub async fn subscribe_sensor(&self, topic: &str) -> anyhow::Result<()> {
        self.mqtt_client.subscribe(topic, QoS::AtLeastOnce).await?;
        Ok(())
    }

    pub fn register_actuator(&mut self, id: &str, topic: &str) {
        self.actuators.insert(id.to_string(), Actuator {
            id: id.to_string(),
            topic: topic.to_string(),
        });
    }

    pub async fn actuate(&self, actuator_id: &str, command: &str) -> anyhow::Result<()> {
        if let Some(actuator) = self.actuators.get(actuator_id) {
            let payload = serde_json::json!({ "command": command }).to_string();
            self.mqtt_client.publish(&actuator.topic, QoS::AtLeastOnce, false, payload).await?;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Atuador {} não encontrado", actuator_id))
        }
    }
}
