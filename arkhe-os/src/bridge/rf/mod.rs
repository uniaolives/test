// arkhe-os/src/bridge/rf/mod.rs
pub mod satellite_bridge;
pub mod ham_radio_bridge;
pub mod wspr_bridge;
pub mod tracking_bridge;
pub mod parallax_bridge;
pub mod wifi_pi_bridge;
pub mod irs_bridge;
pub mod lml_bridge;

#[derive(Debug, Clone)]
pub struct RfFrame {
    pub data: Vec<u8>,
    pub frequency_hz: u64,
    pub modulation: ModulationType,
}

#[derive(Debug, Clone)]
pub enum ModulationType {
    QPSK,
    BPSK,
    QAM16,
    OFDM,
}
