// qhttp_frame.rs
// Frame Format para o protocolo qhttp:// sobre Instaweb

use crate::instaweb_core::HyperbolicCoord;

#[repr(C, packed)]
pub struct QhttpHeader {
    pub magic: [u8; 4],           // b"QHTP"
    pub version: u8,              // 0x01
    pub flags: u8,                // BIT0=emergency, BIT1=ack, BIT2=entanglement
    pub priority: u8,             // 0-255 (255 = Art. 13 override)
    pub payload_len: u16,         // Tamanho em bytes
    pub src_coords: HyperbolicCoord,  // Origem
    pub dst_coords: HyperbolicCoord,  // Destino
    pub timestamp_ns: u64,        // Timestamp de geração (SyncE)
    pub qubit_id: [u8; 16],       // UUID do par EPR
    pub bell_result: u8,          // 00, 01, 10, 11
    pub checksum: u32,            // CRC32
}

pub enum QhttpState {
    EntanglementRequest = 0x01,
    EprReady = 0x02,
    BellMeasurement = 0x03,
    CorrectionApplied = 0x04,
    TeleportComplete = 0x05,
}
