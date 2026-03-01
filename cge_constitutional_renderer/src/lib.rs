pub mod core;
pub mod timing;
pub mod benchmark;

pub use crate::core::constitutional_renderer::*;
pub use crate::timing::constitutional_fps_controller::*;
pub use crate::benchmark::constitutional_benchmark::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constitutional_phi() {
        assert_eq!(phi_calculus::PHI_TARGET, 1.038);
    }
}
