use std::fs::{OpenOptions};
use std::io::Write;
use std::path::Path;
use super::handover::HandoverPacket;

pub struct LedgerStore {
    root_path: String,
}

impl LedgerStore {
    pub fn new() -> Self {
        Self {
            root_path: "/mnt/ledger".to_string(),
        }
    }

    pub fn record(&self, packet: &HandoverPacket) -> anyhow::Result<()> {
        if !Path::new(&self.root_path).exists() {
            std::fs::create_dir_all(&self.root_path)?;
        }

        let ledger_file = Path::new(&self.root_path).join("handover_log.jsonl");
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(ledger_file)?;

        let line = serde_json::to_string(packet)? + "\n";
        file.write_all(line.as_bytes())?;

        tracing::info!("Recorded packet {} to ledger", packet.id);
        Ok(())
    }
}
