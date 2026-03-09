// arkhe-os/src/bridge/rf/mod.rs
pub mod satellite_bridge;
pub mod ham_radio_bridge;

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
