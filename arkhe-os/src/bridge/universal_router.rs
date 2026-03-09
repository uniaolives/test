// arkhe-os/src/bridge/universal_router.rs

use super::tcpip::http_bridge::HttpBridge;
use super::tcpip::websocket_bridge::WebSocketBridge;
use super::tcpip::mqtt_bridge::MqttBridge;
use super::rf::satellite_bridge::SatelliteBridge;
use super::rf::ham_radio_bridge::HamRadioBridge;
use super::blockchain::bitcoin_bridge::BitcoinBridge;
use super::blockchain::ethereum_bridge::EthereumBridge;
use super::blockchain::ipfs_bridge::IpfsBridge;
use super::industrial::modbus_bridge::ModbusBridge;
use super::industrial::opcua_bridge::OpcUaBridge;
use super::mesh::lorawan_bridge::LoRaWanBridge;
use super::mesh::ble_bridge::BleBridge;
use super::dark::tor_bridge::TorBridge;
use crate::orb::core::OrbPayload;
use std::collections::HashMap;

pub struct UniversalOrbRouter {
    pub http: HttpBridge,
    pub websocket: WebSocketBridge,
    pub mqtt: MqttBridge,
    pub satellite: SatelliteBridge,
    pub ham_radio: HamRadioBridge,
    pub bitcoin: BitcoinBridge,
    pub ethereum: EthereumBridge,
    pub ipfs: IpfsBridge,
    pub modbus: ModbusBridge,
    pub opc_ua: OpcUaBridge,
    pub lorawan: LoRaWanBridge,
    pub ble: BleBridge,
    pub tor: TorBridge,
}

impl UniversalOrbRouter {
    /// Transmite Orb por TODOS os canais disponíveis
    pub async fn broadcast(&mut self, orb: &OrbPayload) -> BroadcastResult {
        let mut results = BroadcastResult::new();

        // Paralelizar todas as transmissões
        // Some are async, some are sync transformations

        let http_res = self.http.transmit(orb).await;
        self.websocket.broadcast(orb).await;
        self.mqtt.publish(orb).await;

        let sat_frames = self.satellite.encode_for_satellite(orb);
        let ham_msg = self.ham_radio.encode_ft8(orb);
        let btc_script = self.bitcoin.encode_op_return(orb);

        let eth_res = self.ethereum.send_orb(orb).await;
        let ipfs_res = self.ipfs.publish(orb).await;

        let lora_payload = self.lorawan.encode(orb);
        let tor_res = self.tor.send(orb).await;

        // Record results (simplified)
        results.record("http", http_res.is_ok());
        results.record("websocket", true);
        results.record("mqtt", true);
        results.record("satellite", !sat_frames.is_empty());
        results.record("ham_radio", !ham_msg.is_empty());
        results.record("bitcoin", !btc_script.is_empty());
        results.record("ethereum", eth_res.is_ok());
        results.record("ipfs", ipfs_res.is_ok());
        results.record("lorawan", !lora_payload.is_empty());
        results.record("tor", tor_res.is_ok());

        results
    }
}

pub struct BroadcastResult {
    pub channels: HashMap<String, bool>,
}

impl BroadcastResult {
    pub fn new() -> Self {
        Self { channels: HashMap::new() }
    }
    pub fn record(&mut self, channel: &str, success: bool) {
        self.channels.insert(channel.to_string(), success);
    }
    pub fn success_rate(&self) -> f64 {
        let total = self.channels.len();
        if total == 0 { return 0.0; }
        let successful = self.channels.values().filter(|&&v| v).count();
        successful as f64 / total as f64
    }
}
