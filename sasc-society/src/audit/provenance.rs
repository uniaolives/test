use blake3;

pub struct ProvenanceTracer {
    pub current_hash: blake3::Hash,
}

impl ProvenanceTracer {
    pub fn new() -> Self {
        Self {
            current_hash: blake3::hash(b"SOT_PROVENANCE_ROOT"),
        }
    }

    pub fn trace_round(&mut self, data: &str) {
        let mut hasher = blake3::Hasher::new();
        hasher.update(self.current_hash.as_bytes());
        hasher.update(data.as_bytes());
        self.current_hash = hasher.finalize();
    }
}
