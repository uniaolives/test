pub mod foundry;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    ACTIVE,
    SUSPENDED,
    TERMINATED,
}
