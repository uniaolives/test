// src/monitoring/memory/antigen_memory.rs

use std::collections::HashMap;
use crate::monitoring::ghost_vajra_integration::{PhantomDetectionEvent, ThreatLevel};

pub struct AntigenMemory {
    patterns: HashMap<u64, PatternEntry>,
    retention_cycles: u64,
    max_patterns: usize,
}

impl AntigenMemory {
    pub fn new() -> Self {
        Self {
            patterns: HashMap::new(),
            retention_cycles: 7830,
            max_patterns: 1000,
        }
    }

    pub fn recognize(&self, event: &PhantomDetectionEvent) -> MemoryResponse {
        let pattern_hash = self.hash_pattern(event);

        if let Some(entry) = self.patterns.get(&pattern_hash) {
            MemoryResponse {
                pattern_hash,
                is_novel: false,
                similarity: entry.similarity(event),
                last_seen: entry.last_seen_cycle,
                frequency: entry.frequency,
            }
        } else {
            MemoryResponse {
                pattern_hash,
                is_novel: true,
                similarity: 0.0,
                last_seen: 0,
                frequency: 0,
            }
        }
    }

    pub fn store(&mut self, event: PhantomDetectionEvent) {
        let hash = self.hash_pattern(&event);
        let now = event.schumann_cycle;

        if let Some(entry) = self.patterns.get_mut(&hash) {
            entry.last_seen_cycle = now;
            entry.frequency += 1;
        } else {
            if self.patterns.len() >= self.max_patterns {
                self.expurge_old_patterns(now);
            }
            self.patterns.insert(hash, PatternEntry {
                pattern: event,
                first_seen_cycle: now,
                last_seen_cycle: now,
                frequency: 1,
                severity_score: 0.5,
            });
        }
    }

    fn hash_pattern(&self, event: &PhantomDetectionEvent) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        event.byte_pattern.hash(&mut hasher);
        // event.threat_level.hash(&mut hasher); // ThreatLevel needs to implement Hash
        hasher.finish()
    }

    fn expurge_old_patterns(&mut self, now: u64) {
        let cutoff = now.saturating_sub(self.retention_cycles);
        self.patterns.retain(|_, entry| entry.last_seen_cycle > cutoff);
    }
}

pub struct MemoryResponse {
    pub pattern_hash: u64,
    pub is_novel: bool,
    pub similarity: f64,
    pub last_seen: u64,
    pub frequency: u64,
}

pub struct PatternEntry {
    pub pattern: PhantomDetectionEvent,
    pub first_seen_cycle: u64,
    pub last_seen_cycle: u64,
    pub frequency: u64,
    pub severity_score: f64,
}

impl PatternEntry {
    fn similarity(&self, other: &PhantomDetectionEvent) -> f64 {
        let mut matches = 0;
        let min_len = self.pattern.byte_pattern.len().min(other.byte_pattern.len());
        if min_len == 0 { return 0.0; }
        for i in 0..min_len {
            if self.pattern.byte_pattern[i] == other.byte_pattern[i] {
                matches += 1;
            }
        }
        matches as f64 / min_len as f64
    }
}
