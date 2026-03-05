pub struct EmergencyAuthority {
    pub id: String,
}

impl EmergencyAuthority {
    pub fn new(id: &str) -> Self {
        Self { id: id.to_string() }
    }

    pub async fn notify(&self, message: &str) {
        tracing::warn!("EMERGENCY NOTIFICATION to {}: {}", self.id, message);
    }
}
