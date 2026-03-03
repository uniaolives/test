// metalanguage/rosetta/human_tool/human_tool.rs
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
    Blocked { reason: String, load: f64, time: u64 },
    Generated { load: f64, intent: String, time: u64 },
    Reviewed { approved: bool, output: String, time: u64 },
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

    fn now_ms() -> u64 {
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64
    }

    pub fn propose_interaction(&mut self, intent: &str) -> Option<String> {
        let load = (self.tool.output_volume * self.tool.output_entropy) / self.human.processing_capacity;

        if load > self.threshold {
            self.log.push_back(LogEvent::Blocked {
                reason: "cognitive_overload".into(),
                load,
                time: Self::now_ms(),
            });
            return None;
        }

        if self.human.current_load > 0.8 {
            self.log.push_back(LogEvent::Blocked {
                reason: "human_overloaded".into(),
                load: self.human.current_load,
                time: Self::now_ms(),
            });
            return None;
        }

        let output = format!("Generated content for: {}", intent);

        let impact = load * 0.3;
        self.human.current_load = (self.human.current_load + impact).min(1.0);

        self.log.push_back(LogEvent::Generated {
            load,
            intent: intent.into(),
            time: Self::now_ms(),
        });

        Some(output)
    }

    pub fn review(&mut self, output: &str, approved: bool) {
        self.log.push_back(LogEvent::Reviewed {
            approved,
            output: output.chars().take(100).collect(),
            time: Self::now_ms(),
        });
        if approved {
            self.human.current_load = (self.human.current_load - 0.1).max(0.0);
        }
    }

    pub fn cognitive_load_index(&self, window_minutes: u64) -> f64 {
        let cutoff = Self::now_ms() - window_minutes * 60 * 1000;
        let recent: Vec<_> = self.log.iter().filter(|e| match e {
            LogEvent::Blocked { time, .. } => *time > cutoff,
            LogEvent::Generated { time, .. } => *time > cutoff,
            LogEvent::Reviewed { time, .. } => *time > cutoff,
        }).collect();

        if recent.is_empty() { return 0.0; }

        let overloads = recent.iter().filter(|e| match e {
            LogEvent::Blocked { load, .. } => *load > self.threshold,
            LogEvent::Generated { load, .. } => *load > self.threshold,
            _ => false,
        }).count();

        overloads as f64 / recent.len() as f64
    }

    pub fn authorship_loss_rate(&self, window_minutes: u64) -> f64 {
        let cutoff = Self::now_ms() - window_minutes * 60 * 1000;
        let recent: Vec<_> = self.log.iter().filter(|e| match e {
            LogEvent::Blocked { time, .. } => *time > cutoff,
            LogEvent::Generated { time, .. } => *time > cutoff,
            LogEvent::Reviewed { time, .. } => *time > cutoff,
        }).collect();

        let reviews = recent.iter().filter(|e| matches!(e, LogEvent::Reviewed { .. })).count();
        let total = recent.iter().filter(|e| matches!(e, LogEvent::Generated { .. } | LogEvent::Reviewed { .. })).count();

        if total == 0 { return 0.0; }
        reviews as f64 / total as f64
    }
}
