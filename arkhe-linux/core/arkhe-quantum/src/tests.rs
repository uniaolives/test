#[cfg(test)]
mod tests {
    use crate::safety::rescue_protocol::{RescueProtocol, RescueError};
    use crate::anima_mundi::AnimaMundi;
    use crate::emergency::EmergencyAuthority;
    use crate::constitution::Z3Guard;
    use crate::ledger::OmegaLedger;
    use crate::manifold_ext::ExtendedManifold;
    use arkhe_thermodynamics::InternalModel;
    use std::sync::Arc;
    use tokio::sync::RwLock;

    #[tokio::test]
    async fn test_rescue_protocol_thermalize() {
        let ledger_path = "/tmp/test_ledger_rescue_unique_18";
        let _ = std::fs::remove_dir_all(ledger_path);

        let ledger = Arc::new(OmegaLedger::open(ledger_path).unwrap());

        let core = {
            let ext_manifold = ExtendedManifold::new_with_ledger("localhost", (*ledger).clone()).await.unwrap();
            Arc::new(RwLock::new(AnimaMundi::new(ext_manifold, InternalModel::new())))
        };

        let mut rescue = RescueProtocol::new(
            Arc::new(EmergencyAuthority::new("test-op")),
            Arc::new(Z3Guard),
            ledger.clone(),
            core.clone(),
            None,
        );

        let res: Result<(), RescueError> = rescue.monitor_cycle().await;
        assert!(res.is_ok());

        assert!(ledger.verify_chain().await);
    }
}
