use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::sync::Mutex;
use crate::schema::{Handover, VacuumSnapshot};

pub struct TeknetLedger {
    file: Mutex<BufWriter<File>>,
    memory_index: Mutex<Vec<Handover>>,
}

impl TeknetLedger {
    pub fn new(path: &str) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;

        Ok(Self {
            file: Mutex::new(BufWriter::new(file)),
            memory_index: Mutex::new(Vec::new()),
        })
    }

    pub fn commit_handover(&self, mut handover: Handover) -> std::io::Result<u64> {
        let mut index = self.memory_index.lock().unwrap();
        let id = index.len() as u64;
        handover.id = id;

        let json = serde_json::to_string(&handover).unwrap();

        let mut file = self.file.lock().unwrap();
        writeln!(file, "{}", json)?;
        file.flush()?;

        index.push(handover.clone());

        Ok(id)
    }

    pub fn commit_snapshot(&self, snapshot: VacuumSnapshot) -> std::io::Result<()> {
        let json = serde_json::to_string(&snapshot).unwrap();
        let mut file = self.file.lock().unwrap();
        writeln!(file, "SNAPSHOT: {}", json)?;
        file.flush()?;
        Ok(())
    }

    pub fn get_history(&self) -> Vec<Handover> {
        let index = self.memory_index.lock().unwrap();
        index.clone()
    }
}
