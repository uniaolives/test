use std::cmp::Ordering;

#[derive(Debug, Clone)]
pub struct Task {
    pub id: u64,
    pub name: String,
    pub coherence_required: f64,
    pub estimated_duration: u64,
    pub priority: i32,
    pub created_at: std::time::Instant,
    pub time_consumed: u64,
}

impl Task {
    pub fn new(id: u64, name: &str, coherence: f64, duration: u64, priority: i32) -> Self {
        Self {
            id,
            name: name.to_string(),
            coherence_required: coherence,
            estimated_duration: duration,
            priority,
            created_at: std::time::Instant::now(),
            time_consumed: 0,
        }
    }

    pub fn required_density(&self) -> f64 {
        crate::lib::miller::coherence_to_density(self.coherence_required)
    }
}

impl Ord for Task {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .cmp(&other.priority)
            .then_with(|| other.estimated_duration.cmp(&self.estimated_duration))
    }
}

impl PartialOrd for Task {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Task {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Task {}
