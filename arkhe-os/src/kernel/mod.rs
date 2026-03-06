//! Módulo principal do kernel.

pub mod task;
pub mod allocator;
pub mod scheduler;
pub mod syscall;

pub use task::Task;
pub use allocator::CoherenceAllocator;
pub use scheduler::CoherenceScheduler;
pub use syscall::{SyscallHandler, SyscallResult};
