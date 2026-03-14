// arkhe-os/src/bridge/blockchain/bitcoin_bridge.rs

use bitcoin::blockdata::script::ScriptBuf;
use bitcoin::script::PushBytes;
use crate::orb::core::OrbPayload;
use crc::Crc;
use std::convert::TryInto;

pub struct BitcoinBridge {
    _network: bitcoin::Network,
}

impl BitcoinBridge {
    pub fn new(network: bitcoin::Network) -> Self {
        Self { _network: network }
    }

    /// Codifica Orb em OP_RETURN (máx 80 bytes)
    pub fn encode_op_return(&self, orb: &OrbPayload) -> ScriptBuf {
        let mut data = Vec::new();
        let crc_algo = Crc::<u32>::new(&crc::CRC_32_ISO_HDLC);

        // Magic bytes para identificar Orb
        data.extend_from_slice(b"ORB:");

        // Hash do Orb (32 bytes)
        data.extend_from_slice(&orb.orb_id);

        // Lambda_2 comprimido (2 bytes)
        let lambda_compressed = (orb.lambda_2 * 1000.0) as u16;
        data.extend_from_slice(&lambda_compressed.to_be_bytes());

        // Phi_q comprimido (2 bytes)
        let phi_compressed = (orb.phi_q * 100.0) as u16;
        data.extend_from_slice(&phi_compressed.to_be_bytes());

        // Timestamp (4 bytes)
        data.extend_from_slice(&(orb.origin_time as u32).to_be_bytes());

        // Target time (4 bytes)
        data.extend_from_slice(&(orb.target_time as u32).to_be_bytes());

        // CRC (4 bytes)
        let crc_val = crc_algo.checksum(&data);
        data.extend_from_slice(&crc_val.to_be_bytes());

        // OP_RETURN script
        let push_bytes: &PushBytes = data.as_slice().try_into().unwrap();
        bitcoin::blockdata::script::Builder::new()
            .push_opcode(bitcoin::blockdata::opcodes::all::OP_RETURN)
            .push_slice(push_bytes)
            .into_script()
    }
}
