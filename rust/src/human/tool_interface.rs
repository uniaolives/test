// rust/src/human/tool_interface.rs
// Arkhe(n) Human-Tool Interface Core Implementation

use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct Human {
    pub processing_capacity: f64,  // bits/min
    pub attention_span: f64,        // minutes
    pub current_load: f64,
    pub goals: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Tool {
    pub output_volume: f64,         // tokens/min
    pub output_entropy: f64,        // bits/token
    pub has_discernment: bool,
    pub has_intentionality: bool,
    pub has_perception: bool,
}

#[derive(Debug)]
pub enum LogEvent {
    Blocked { timestamp: u64, reason: String, load: f64 },
    Generated { timestamp: u64, load: f64, intent: String },
    Reviewed { timestamp: u64, approved: bool, output: String },
}

impl LogEvent {
    pub fn timestamp(&self) -> u64 {
        match self {
            LogEvent::Blocked { timestamp, .. } => *timestamp,
            LogEvent::Generated { timestamp, .. } => *timestamp,
            LogEvent::Reviewed { timestamp, .. } => *timestamp,
        }
    }
}

pub struct InteractionGuard {
    pub human: Human,
    pub tool: Tool,
    pub log: VecDeque<LogEvent>,
    pub threshold: f64,
}

impl InteractionGuard {
    pub fn new(human: Human, tool: Tool) -> Self {
        Self {
            human,
            tool,
            log: VecDeque::new(),
            threshold: 0.7,
        }
    }

    fn now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    pub fn propose_interaction(&mut self, intent: &str) -> Option<String> {
        let load = (self.tool.output_volume * self.tool.output_entropy) / self.human.processing_capacity;
        let timestamp = Self::now();

        if load > self.threshold {
            self.log.push_back(LogEvent::Blocked {
                timestamp,
                reason: "cognitive_overload".into(),
                load,
            });
            return None;
        }

        if self.human.current_load > 0.8 {
            self.log.push_back(LogEvent::Blocked {
                timestamp,
                reason: "human_overloaded".into(),
                load: self.human.current_load,
            });
            return None;
        }

        // Simulate generation
        let output = format!("Generated content for: {}", intent);

        let impact = load * 0.3;
        self.human.current_load = (self.human.current_load + impact).min(1.0);

        self.log.push_back(LogEvent::Generated {
            timestamp,
            load,
            intent: intent.into(),
        });

        Some(output)
    }

    pub fn review(&mut self, output: &str, approved: bool) {
        self.log.push_back(LogEvent::Reviewed {
            timestamp: Self::now(),
            approved,
            output: output.chars().take(100).collect(),
        });
        if approved {
            self.human.current_load = (self.human.current_load - 0.1).max(0.0);
        }
    }

    pub fn cognitive_load_index(&self, window_seconds: u64) -> f64 {
        let cutoff = Self::now().saturating_sub(window_seconds);

        let recent: Vec<_> = self.log
            .iter()
            .filter(|e| e.timestamp() >= cutoff)
            .filter(|e| match e {
                LogEvent::Blocked { .. } => true,
                LogEvent::Generated { .. } => true,
                LogEvent::Reviewed { .. } => false,
            })
            .collect();

        if recent.is_empty() { return 0.0; }

        let overloads = recent.iter()
            .filter(|e| match e {
                LogEvent::Blocked { load, .. } => *load > self.threshold,
                LogEvent::Generated { load, .. } => *load > self.threshold,
                _ => false,
            })
            .count();

        overloads as f64 / recent.len() as f64
    }

    pub fn authorship_loss_rate(&self, window_seconds: u64) -> f64 {
        let cutoff = Self::now().saturating_sub(window_seconds);

        let recent: Vec<_> = self.log
            .iter()
            .filter(|e| e.timestamp() >= cutoff)
            .collect();

        if recent.is_empty() { return 0.0; }

        let reviews = recent.iter()
            .filter(|e| matches!(e, LogEvent::Reviewed { .. }))
            .count();

        let total = recent.len();

        reviews as f64 / total as f64
    }
}
