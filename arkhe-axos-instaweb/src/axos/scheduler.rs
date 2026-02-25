// arkhe-axos-instaweb/src/axos/scheduler.rs
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use crate::axos::integrity_gates::{Operation, IntegrityGates, IntegrityError};

#[derive(Debug, Clone)]
pub struct ScheduledTask {
    pub priority: u32,
    pub operation: Operation,
}

impl PartialEq for ScheduledTask {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for ScheduledTask {}

impl Ord for ScheduledTask {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.cmp(&other.priority)
    }
}

impl PartialOrd for ScheduledTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct AxosScheduler {
    pub task_queue: BinaryHeap<ScheduledTask>,
}

impl AxosScheduler {
    pub fn new() -> Self {
        Self {
            task_queue: BinaryHeap::new(),
        }
    }

    pub fn schedule(&mut self, priority: u32, operation: Operation) {
        self.task_queue.push(ScheduledTask { priority, operation });
    }

    pub fn run_next(&mut self) -> Result<String, String> {
        if let Some(task) = self.task_queue.pop() {
            match IntegrityGates::verify(&task.operation) {
                Ok(_) => Ok(format!("Executed: {}", task.operation.id)),
                Err(e) => {
                    let err_msg = match e {
                        IntegrityError::ConservationViolation(v) => format!("Conservation Violation: {}", v),
                        IntegrityError::CriticalityViolation(v) => format!("Criticality Violation: {}", v),
                        IntegrityError::UnauthorizedCriticalOperation => "Unauthorized Critical Operation".to_string(),
                        IntegrityError::FailClosedTriggered(s) => format!("Fail-Closed: {}", s),
                    };
                    Err(err_msg)
                }
            }
        } else {
            Err("No tasks in queue".to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arkhe::invariants::ArkheState;
    use crate::axos::integrity_gates::CapabilityLevel;
    use rust_decimal_macros::dec;

    #[test]
    fn test_scheduling_and_execution() {
        let mut scheduler = AxosScheduler::new();
        let op = Operation {
            id: "task1".to_string(),
            capability: CapabilityLevel::Basic,
            proposed_state: ArkheState::new(dec!(0.5), dec!(0.5), 0.1),
            requires_human_approval: false,
            is_human_approved: false,
        };
        scheduler.schedule(10, op);
        let res = scheduler.run_next();
        assert_eq!(res.unwrap(), "Executed: task1");
    }

    #[test]
    fn test_scheduling_priority() {
        let mut scheduler = AxosScheduler::new();
        let op1 = Operation {
            id: "low".to_string(),
            capability: CapabilityLevel::Basic,
            proposed_state: ArkheState::new(dec!(0.5), dec!(0.5), 0.1),
            requires_human_approval: false,
            is_human_approved: false,
        };
        let op2 = Operation {
            id: "high".to_string(),
            capability: CapabilityLevel::Basic,
            proposed_state: ArkheState::new(dec!(0.5), dec!(0.5), 0.1),
            requires_human_approval: false,
            is_human_approved: false,
        };
        scheduler.schedule(1, op1);
        scheduler.schedule(100, op2);

        assert_eq!(scheduler.run_next().unwrap(), "Executed: high");
        assert_eq!(scheduler.run_next().unwrap(), "Executed: low");
    }
}
