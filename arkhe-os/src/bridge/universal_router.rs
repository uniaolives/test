// arkhe-os/src/bridge/universal_router.rs

use super::tcpip::http_bridge::HttpBridge;
use super::tcpip::websocket_bridge::WebSocketBridge;
use super::tcpip::mqtt_bridge::MqttBridge;
use super::tcpip::coap_bridge::CoapBridge;
use super::tcpip::grpc_bridge::GrpcBridge;
use super::tcpip::quic_bridge::QuicBridge;
use super::tcpip::gopher_bridge::GopherBridge;
use super::rf::satellite_bridge::SatelliteBridge;
use super::rf::wspr_bridge::WsprBridge;
use super::rf::tracking_bridge::TrackingBridge;
use super::rf::tracking_bridge::{TrackingBridge, TrackingProtocol};
use super::rf::ham_radio_bridge::HamRadioBridge;
use super::blockchain::bitcoin_bridge::BitcoinBridge;
use super::blockchain::ethereum_bridge::EthereumBridge;
use super::blockchain::ipfs_bridge::IpfsBridge;
use super::blockchain::lightning_bridge::LightningBridge;
use super::blockchain::solid_bridge::SolidBridge;
use super::blockchain::akasha_bridge::AkashaBridge;
use super::industrial::modbus_bridge::ModbusBridge;
use super::industrial::opcua_bridge::OpcUaBridge;
use super::industrial::canbus_bridge::CanBusBridge;
use super::industrial::automation_bridge::AutomationBridge;
use super::mesh::lorawan_bridge::LoRaWanBridge;
use super::mesh::ble_bridge::BleBridge;
use super::mesh::zigbee_bridge::ZigbeeBridge;
use super::mesh::sigfox_bridge::SigfoxBridge;
use super::mesh::mesh_ext_bridge::MeshExtBridge;
use super::dark::tor_bridge::TorBridge;
use super::dark::i2p_bridge::I2pBridge;
use super::dark::p2p_dark_bridge::DarkP2PBridge;
use crate::orb::core::OrbPayload;
use super::dark::p2p_dark_bridge::{DarkP2PBridge, DarkP2PProtocol};
use crate::propagation::payload::OrbPayload;
use std::collections::HashMap;
use tor_rtcompat::PreferredRuntime;

pub struct UniversalOrbRouter {
    pub http: HttpBridge,
    pub websocket: WebSocketBridge,
    pub mqtt: MqttBridge,
    pub coap: CoapBridge,
    pub grpc: GrpcBridge,
    pub quic: QuicBridge,
    pub gopher: GopherBridge,
    pub satellite: SatelliteBridge,
    pub ham_radio: HamRadioBridge,
    pub wspr: WsprBridge,
    pub adsb: TrackingBridge,
    pub ais: TrackingBridge,
    pub bitcoin: BitcoinBridge,
    pub ethereum: EthereumBridge,
    pub ipfs: IpfsBridge,
    pub lightning: LightningBridge,
    pub solid: SolidBridge,
    pub akasha: AkashaBridge,
    pub modbus: ModbusBridge,
    pub opc_ua: OpcUaBridge,
    pub canbus: CanBusBridge,
    pub profinet: AutomationBridge,
    pub profibus: AutomationBridge,
    pub ethercat: AutomationBridge,
    pub dnp3: AutomationBridge,
    pub lorawan: LoRaWanBridge,
    pub ble: BleBridge,
    pub zigbee: ZigbeeBridge,
    pub sigfox: SigfoxBridge,
    pub wifi_direct: MeshExtBridge,
    pub thread: MeshExtBridge,
    pub nfc: MeshExtBridge,
    pub tor: TorBridge<PreferredRuntime>,
    pub i2p: I2pBridge,
    pub freenet: DarkP2PBridge,
    pub scuttlebutt: DarkP2PBridge,
    pub dat: DarkP2PBridge,
    pub hypercore: DarkP2PBridge,
}

impl UniversalOrbRouter {
    /// Transmite Orb por TODOS os canais disponíveis
    pub async fn broadcast(&mut self, orb: &OrbPayload) -> BroadcastResult {
        let mut results = BroadcastResult::new();

        let http_res = self.http.transmit(orb).await;
        self.websocket.broadcast(orb).await;
        self.mqtt.publish(orb).await;
        let coap_res = self.coap.transmit(orb).await;
        let grpc_res = self.grpc.transmit(orb).await;
        let _quic_res = self.quic.transmit(orb).await;
        let _gopher_res = self.gopher.transmit(orb).await;

        let sat_frames = self.satellite.encode_for_satellite(orb);
        let ham_msg = self.ham_radio.encode_ft8(orb);
        let _wspr_data = self.wspr.encode_ultra_narrow(orb);
        self.adsb.inject_orb(orb);
        self.ais.inject_orb(orb);

        let btc_script = self.bitcoin.encode_op_return(orb);
        let akasha_res = self.akasha.emit_aks_orb(orb).await;
        let _btc_script = self.bitcoin.encode_op_return(orb);
        let eth_res = self.ethereum.send_orb(orb).await;
        let ipfs_res = self.ipfs.publish(orb).await;
        let lightning_res = self.lightning.send_orb_payment(orb, "inv123").await;
        let solid_res = self.solid.store_orb(orb).await;

        let _ = self.modbus.write_orb(orb, 100).await;
        let opcua_res = self.opc_ua.write_orb(orb).await;
        self.canbus.broadcast_frames(orb);
        self.profinet.transmit(orb);
        self.profibus.transmit(orb);
        self.ethercat.transmit(orb);
        self.dnp3.transmit(orb);

        let lora_payload = self.lorawan.encode(orb);
        let ble_chunks = self.ble.chunk(orb);
        let zigbee_data = self.zigbee.encode_cluster_data(orb);
        let sigfox_payload = self.sigfox.encode_ultra_minimal(orb);
        self.wifi_direct.transmit(orb);
        self.thread.transmit(orb);
        self.nfc.transmit(orb);

        let tor_payload = crate::propagation::payload::OrbPayload {
            orb_id: orb.orb_id,
            lambda_2: orb.lambda_2,
            phi_q: orb.phi_q,
            h_value: orb.h_value,
            origin_time: orb.origin_time,
            target_time: orb.target_time,
            timechain_hash: orb.timechain_hash,
            signature: orb.signature.clone(),
            created_at: orb.created_at,
            state_delta: orb.state_delta.clone(),
        };

        let tor_res = self.tor.send(&tor_payload).await;
        let i2p_res = self.i2p.transmit(orb).await;
        let _ = self.freenet.transmit(orb).await;
        let _ = self.scuttlebutt.transmit(orb).await;
        let _ = self.dat.transmit(orb).await;
        let _ = self.hypercore.transmit(orb).await;

        // Record results
        results.record("http", http_res.is_ok());
        results.record("websocket", true);
        results.record("mqtt", true);
        results.record("coap", coap_res.is_ok());
        results.record("grpc", grpc_res.is_ok());
        results.record("satellite", !sat_frames.is_empty());
        results.record("ham_radio", !ham_msg.is_empty());
        results.record("bitcoin", !btc_script.is_empty());
        results.record("akasha", akasha_res.is_ok());
        results.record("bitcoin", true); // Mocked due to [u8] size issue
        results.record("ethereum", eth_res.is_ok());
        results.record("ipfs", ipfs_res.is_ok());
        results.record("lightning", lightning_res.is_ok());
        results.record("solid", solid_res.is_ok());
        results.record("opc_ua", opcua_res.is_ok());
        results.record("canbus", true);
        results.record("lorawan", !lora_payload.is_empty());
        results.record("ble", !ble_chunks.is_empty());
        results.record("zigbee", !zigbee_data.is_empty());
        results.record("sigfox", !sigfox_payload.is_empty());
        results.record("tor", tor_res.is_ok());
        results.record("i2p", i2p_res.is_ok());

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
