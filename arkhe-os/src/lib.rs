pub mod kernel;
pub mod physics;
pub mod compiler;
pub mod db;
pub mod net;
pub mod phys;
pub mod telemetry;
pub mod sensors;
pub mod intention;
pub mod zkproof;

pub use kernel::task::Task;
pub use kernel::scheduler::{CoherenceScheduler, SchedulerEvent};
pub use physics::miller::{PHI_Q, check_nucleation, quantum_interest};
pub use physics::berry;
pub use compiler::parser::parse_intention_block;
pub use db::ledger::TeknetLedger;
pub use net::node::P2PNode;
pub use phys::ibm_client::QuantumAntenna;

#[cfg(test)]
mod tests;
