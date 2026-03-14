// arkhe-os/src/bridge/tcpip/mqtt_bridge.rs

use rumqttc::{MqttOptions, Client, QoS, Connection};
use crate::propagation::payload::OrbPayload;
use rumqttc::{MqttOptions, Connection, QoS, Event, Incoming};
use crate::orb::core::OrbPayload;

pub struct MqttBridge {
    client: rumqttc::Client,
}

impl MqttBridge {
    pub fn new(broker: &str, port: u16) -> Self {
        let mut mqttoptions = MqttOptions::new("arkhe-orb-node", broker, port);
        mqttoptions.set_keep_alive(std::time::Duration::from_secs(5));

        let (client, mut connection) = rumqttc::Client::new(mqttoptions, 10);

        // Spawn connection loop
        tokio::spawn(async move {
            while let Ok(notification) = connection.eventloop.poll().await {
                if let rumqttc::Event::Incoming(rumqttc::Incoming::Publish(publish)) = notification {
                    if let Ok(orb) = OrbPayload::from_bytes(&publish.payload) {
                        println!("[MQTT] Received Orb: {:?}", orb.orb_id);
            for notification in connection.iter() {
                match notification {
                    Ok(Event::Incoming(Incoming::Publish(publish))) => {
                        if let Ok(orb) = OrbPayload::from_bytes(&publish.payload) {
                            println!("[MQTT] Received Orb: {:?}", orb.orb_id);
                        }
                    }
                    Ok(_) => {}
                    Err(e) => {
                        eprintln!("[MQTT] Connection error: {:?}", e);
                        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                    }
                }
            }
        });

        Self { client }
    }

    /// Publica Orb em múltiplos tópicos
    pub async fn publish(&mut self, orb: &OrbPayload) {
        let topics = vec![
            "arkhe/orb/broadcast",
            "arkhe/temporal/handover",
            "arkhe/retrocausal/channel",
        ];

        let payload = orb.to_bytes();

        for topic in topics {
            let _ = self.client.publish(
                topic,
                QoS::ExactlyOnce,
                false,
                payload.clone(),
            );
        }
    }
}
