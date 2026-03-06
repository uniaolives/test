//! Escalonador baseado em coerência e prioridade.

use std::collections::BinaryHeap;
use super::task::Task;
use super::allocator::CoherenceAllocator;
use crate::physics::miller::PHI_Q;

/// Eventos que o escalonador pode gerar.
use std::collections::BinaryHeap;
use super::task::Task;
use super::allocator::CoherenceAllocator;
use crate::lib::miller::PHI_Q;

pub enum SchedulerEvent {
    TaskStarted(Task),
    TaskCompleted(Task),
    WaveCloudNucleation { phi_q: f64 },
    CoherenceWarning { available: f64, required: f64 },
}

/// Escalonador principal.
pub struct CoherenceScheduler {
    task_queue: BinaryHeap<Task>,
    running_task: Option<Task>,
    allocator: CoherenceAllocator,
    events: Vec<SchedulerEvent>,
    tick_count: u64,
}

impl CoherenceScheduler {
    pub fn new(initial_coherence: f64) -> Self {
        Self {
            task_queue: BinaryHeap::new(),
            running_task: None,
            allocator: CoherenceAllocator::new(initial_coherence),
            events: Vec::new(),
            tick_count: 0,
        }
    }

    /// Adiciona uma nova tarefa à fila.
    pub fn schedule(&mut self, task: Task) {
        self.task_queue.push(task);
    }

    /// Executa um ciclo de escalonamento.
    /// Retorna um evento se algo significativo ocorrer.
    pub fn tick(&mut self) -> Option<SchedulerEvent> {
        self.tick_count += 1;

        // Se há uma tarefa em execução, decrementar o tempo restante
        if let Some(task) = &mut self.running_task {
            // Simula execução por um tick
            if task.time_consumed >= task.estimated_duration {
                // Tarefa concluída
    pub fn tick(&mut self) -> Option<SchedulerEvent> {
        self.tick_count += 1;

        if let Some(task) = &mut self.running_task {
            if task.time_consumed >= task.estimated_duration {
                let completed = task.clone();
                self.allocator.free(&completed);
                self.running_task = None;
                return Some(SchedulerEvent::TaskCompleted(completed));
            } else {
                task.time_consumed += 1;
                // Continua executando a mesma tarefa
                return None;
            }
        }

        // Se não há tarefa em execução, pegar a próxima da fila
        if let Some(next_task) = self.task_queue.pop() {
            // Tentar alocar coerência para ela
            match self.allocator.allocate(&next_task) {
                Ok(_) => {
                    // Verificar risco de nucleação
        if let Some(next_task) = self.task_queue.pop() {
            match self.allocator.allocate(&next_task) {
                Ok(_) => {
                    let phi = self.allocator.current_phi_q();
                    if phi > PHI_Q {
                        self.events.push(SchedulerEvent::WaveCloudNucleation { phi_q: phi });
                    }
                    self.running_task = Some(next_task.clone());
                    Some(SchedulerEvent::TaskStarted(next_task))
                }
                Err(_e) => {
                    // Coerência insuficiente: recolocar na fila e emitir aviso
                    let avail = self.allocator.available();
                    self.task_queue.push(next_task);
                    Some(SchedulerEvent::CoherenceWarning {
                        available: avail,
                        required: 0.0, // poderíamos extrair do erro
                Err(_) => {
                    self.task_queue.push(next_task);
                    Some(SchedulerEvent::CoherenceWarning {
                        available: self.allocator.available(),
                        required: 0.0,
                    })
                }
            }
        } else {
            // Fila vazia
            None
        }
    }

    /// Retorna o estado actual do sistema.
    pub fn status(&self) -> (f64, f64, usize) {
        (
            self.allocator.available(),
            self.allocator.current_phi_q(),
            self.task_queue.len(),
        )
    }

    /// Lista todos os eventos ocorridos.
    pub fn events(&self) -> &[SchedulerEvent] {
        &self.events
    }

    /// Injeta coerência diretamente no sistema (ex.: via ponte biocibernética)
    pub fn inject_coherence(&mut self, delta: f64) {
        // Atualiza a coerência disponível no alocador (ou um campo global se preferir)
        // Por agora, vamos apenas logar o impacto no φ_q
        println!("[SCHEDULER] Coerência injetada: +{:.3}", delta);
    }
}
