pub struct HardwareAttestation {
    node_id: String,
}

impl HardwareAttestation {
    pub fn new(node_id: &str) -> Self {
        Self { node_id: node_id.to_string() }
    }
    pub fn node_id(&self) -> &str {
        &self.node_id
    }
}
