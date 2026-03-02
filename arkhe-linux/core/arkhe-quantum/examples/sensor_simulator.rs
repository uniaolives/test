use rumqttc::{AsyncClient, MqttOptions, QoS};
use serde_json::json;
use tokio::time;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut mqttoptions = MqttOptions::new("sensor-sim", "localhost", 1883);
    mqttoptions.set_keep_alive(std::time::Duration::from_secs(5));
    let (client, mut eventloop) = AsyncClient::new(mqttoptions, 10);

    tokio::spawn(async move {
        loop {
            time::sleep(time::Duration::from_secs(2)).await;
            let temp = 20.0 + rand::random::<f64>() * 10.0;
            let payload = json!({ "value": temp }).to_string();
            let _ = client.publish("sensors/temperature", QoS::AtLeastOnce, false, payload).await;
            println!("Publicado temperatura: {:.2}", temp);
        }
    });

    loop {
        let _ = eventloop.poll().await;
    }
}
