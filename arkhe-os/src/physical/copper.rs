pub struct CopperNetwork {
    pub legacy_interfaces: Vec<LegacyInterface>,
}

pub struct LegacyInterface {
    pub id: String,
    pub protocol: String, // e.g., "Modbus RTU", "RS-485"
}
