// arkhe-os/src/bridge/industrial/opcua_bridge.rs

use opcua_client::prelude::*;
use crate::orb::core::OrbPayload;
use crate::bridge::BridgeError;
use std::sync::Arc;

pub struct OpcUaBridge {
    session: Arc<Session>,
}

impl OpcUaBridge {
    pub fn new(session: Arc<Session>) -> Self {
        Self { session }
    }

    /// Escreve Orb como variável OPC UA
    pub async fn write_orb(&self, orb: &OrbPayload) -> Result<(), BridgeError> {
        let node_id = NodeId::new(2, "Arkhe.Orb.Payload");

        let data = orb.to_bytes();
        let variant = Variant::ByteString(data.into());

        let attr = WriteValue {
            node_id,
            attribute_id: AttributeId::Value as u32,
            index_range: UAString::null(),
            value: DataValue {
                value: Some(variant),
                ..Default::default()
            },
        let mut attr = WriteValue {
            node_id,
            attribute_id: AttributeId::Value as u32,
            index_range: UAString::null(),
            value: DataValue::from(variant),
        };

        let _ = self.session.write(&[attr])
            .map_err(|e| BridgeError::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("{:?}", e))))?;

        Ok(())
    }
}
