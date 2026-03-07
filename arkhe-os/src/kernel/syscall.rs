//! Chamadas de sistema (syscalls) para interacção com o kernel.

use super::task::Task;
use super::scheduler::{CoherenceScheduler, SchedulerEvent};
use crate::physics::miller::quantum_interest;

/// Resultado de uma syscall.
pub enum SyscallResult {
    Success(String),
    Error(String),
    CoherenceUpdate(f64),
    TaskId(u64),
    WaveCloudStatus(bool, f64),
}

/// Implementa as syscalls disponíveis para programas de usuário.
pub struct SyscallHandler {
    scheduler: CoherenceScheduler,
    next_task_id: u64,
}

impl SyscallHandler {
    pub fn new(initial_coherence: f64) -> Self {
        Self {
            scheduler: CoherenceScheduler::new(initial_coherence),
            next_task_id: 1,
        }
    }

    /// Cria uma nova tarefa.
    pub fn sys_create_task(&mut self, name: &str, coherence: f64, duration: u64, priority: i32) -> SyscallResult {
        let task = Task::new(self.next_task_id, name, coherence, duration, priority);
        self.next_task_id += 1;
        self.scheduler.schedule(task.clone());
        SyscallResult::TaskId(task.id)
    }

    /// Executa um ciclo de escalonamento.
    pub fn sys_tick(&mut self) -> SyscallResult {
        if let Some(event) = self.scheduler.tick() {
            match event {
                SchedulerEvent::TaskStarted(t) => SyscallResult::Success(format!("Task {} started", t.id)),
                SchedulerEvent::TaskCompleted(t) => SyscallResult::Success(format!("Task {} completed", t.id)),
                SchedulerEvent::WaveCloudNucleation { phi_q } => SyscallResult::Success(format!("WAVE-CLOUD NUCLEATION at φ_q = {:.3}", phi_q)),
                SchedulerEvent::CoherenceWarning { available, .. } => SyscallResult::Error(format!("Coherence low: {:.3} available", available)),
            }
        } else {
            SyscallResult::Success("Idle".to_string())
        }
    }

    /// Consulta o estado actual da coerência.
    pub fn sys_coherence_status(&mut self) -> SyscallResult {
        let (avail, _, _) = self.scheduler.status();
        SyscallResult::CoherenceUpdate(avail)
    }

    /// Verifica se o limiar de Miller foi ultrapassado.
    pub fn sys_check_nucleation(&mut self) -> SyscallResult {
        let (_, phi_q, _) = self.scheduler.status();
        SyscallResult::WaveCloudStatus(phi_q > 4.64, phi_q)
    }

    /// Executa um handover simbólico (teste).
    pub fn sys_handover(&mut self, target_epoch: u32, payload: &str) -> SyscallResult {
        let interest = quantum_interest(0.5, 1.0); // valores fictícios
        SyscallResult::Success(format!("Handover to {} sent with payload '{}'. Interest: {:.3}", target_epoch, payload, interest))
    }
}
