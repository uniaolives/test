pub mod sealer_client {
    tonic::include_proto!("karnak");
}

pub use sealer_client::*;

pub enum Priority {
    Normal = 0,
    High = 1,
    Critical = 2,
}

pub struct SealRequest {
    pub alert: Option<vajra_entropy_monitor::OverloadAlert>,
    pub priority: i32,
}
