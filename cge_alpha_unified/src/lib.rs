pub mod unified_core;
pub mod shader;

#[cfg(test)]
mod tests {
    use super::unified_core::vmcore_orchestrator::UnifiedVMCoreOrchestrator;
    use super::unified_core::UnifiedConfig;

    #[tokio::test]
    async fn test_constitutional_verification() {
        let config = UnifiedConfig::default();
        let result = UnifiedVMCoreOrchestrator::bootstrap(Some(config)).await;
        assert!(result.is_ok(), "Bootstrap should succeed with default config");

        let unified = result.unwrap();
        let phi = *unified.current_phi.read();
        let expected_phi = 1.038_f64.powi(40);

        assert!((phi - expected_phi).abs() < 0.001, "Phi should be within constitutional tolerance");
        assert_eq!(unified.config.total_frags, 113);
        assert_eq!(unified.config.dispatch_bars, 92);
    }
}
