use rumqttc::{AsyncClient, MqttOptions, QoS};
use serde_json::Value;
use tokio::sync::mpsc;
use std::collections::HashMap;
use anyhow::{Result, anyhow};
use log::info;

/// Representa um sensor físico que envia dados.
#[derive(Debug, Clone)]
pub struct Sensor {
    pub id: String,
    pub topic: String,
    pub last_value: f64,
}

/// Representa um atuador físico.
#[derive(Debug, Clone)]
pub struct Actuator {
    pub id: String,
    pub topic: String,
}

/// Evento gerado por um sensor.
#[derive(Debug)]
pub struct SensorEvent {
    pub sensor_id: String,
    pub value: f64,
    pub timestamp: std::time::Instant,
}

/// Gateway DePIN: gerencia sensores e atuadores via MQTT.
pub struct DePinGateway {
    mqtt_client: AsyncClient,
    #[allow(dead_code)]
    event_tx: mpsc::UnboundedSender<SensorEvent>,
    actuators: HashMap<String, Actuator>,
}

impl DePinGateway {
    pub async fn new_with_receiver(broker: &str, port: u16, client_id: &str) -> Result<(Self, mpsc::UnboundedReceiver<SensorEvent>)> {
        let mut mqttoptions = MqttOptions::new(client_id, broker, port);
        mqttoptions.set_keep_alive(std::time::Duration::from_secs(5));
        let (client, mut eventloop) = AsyncClient::new(mqttoptions, 10);

        let (event_tx, event_rx) = mpsc::unbounded_channel();
        let event_tx_clone = event_tx.clone();

        tokio::spawn(async move {
            while let Ok(notification) = eventloop.poll().await {
                if let rumqttc::Event::Incoming(rumqttc::Packet::Publish(publish)) = notification {
                    let topic = publish.topic;
                    let payload = publish.payload;
                    if let Ok(value) = serde_json::from_slice::<Value>(&payload) {
                        if let Some(num) = value.get("value").and_then(|v| v.as_f64()) {
                            let sensor_id = topic.replace('/', "_");
                            let _ = event_tx_clone.send(SensorEvent {
                                sensor_id,
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

    /// Inscreve-se em um tópico de sensor.
    pub async fn subscribe_sensor(&self, topic: &str) -> Result<()> {
        self.mqtt_client.subscribe(topic, QoS::AtLeastOnce).await?;
        Ok(())
    }

    /// Registra um atuador (para envio de comandos).
    pub fn register_actuator(&mut self, id: &str, topic: &str) {
        self.actuators.insert(id.to_string(), Actuator {
            id: id.to_string(),
            topic: topic.to_string(),
        });
    }

    /// Envia comando para um atuador.
    pub async fn actuate(&self, actuator_id: &str, command: &str) -> Result<()> {
        if let Some(actuator) = self.actuators.get(actuator_id) {
            let payload = serde_json::json!({ "command": command }).to_string();
            self.mqtt_client.publish(&actuator.topic, QoS::AtLeastOnce, false, payload).await?;
            info!("Comando enviado para {}: {}", actuator_id, command);
            Ok(())
        } else {
            Err(anyhow!("Atuador {} não encontrado", actuator_id))
        }
    }
}
