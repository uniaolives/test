pub struct HandoverManager;

impl HandoverManager {
    pub fn new() -> Self {
        Self
    }

    pub async fn process_queue(&self) -> Result<(), anyhow::Error> {
        Ok(())
    }

    pub async fn broadcast_phi(&self, _phi: f64) {
    }

    pub async fn send_system_notification(&self, _msg: &str) -> Result<(), anyhow::Error> {
        Ok(())
    }
}
