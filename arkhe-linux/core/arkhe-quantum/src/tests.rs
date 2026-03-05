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
    use crate::oloid::OloidState;
    use crate::chronos::{TimechainAnchor, sanity::RealityAnchor};
    use bitcoin::block::{Header as BlockHeader, BlockHash};
    use bitcoin::hashes::Hash;
    use bitcoin::consensus::encode::deserialize;
    use hex;
    use log::info;

    #[test]
    fn test_timechain_sync() {
        let oloid = OloidState::new();
        let mut anchor = TimechainAnchor::new(oloid);

        // Simulated Bitcoin Block Header (Mainnet Genesis for example)
        // 01000000 - version 1
        // 0000000000000000000000000000000000000000000000000000000000000000 - prev block hash
        // 3ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a - merkle root
        // 29ab5f49 - timestamp
        // ffff001d - bits
        // 1dac2b7c - nonce
        let header_hex = "0100000000000000000000000000000000000000000000000000000000000000000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a29ab5f49ffff001d1dac2b7c";
        let header_bytes = hex::decode(header_hex).unwrap();
        let header: BlockHeader = deserialize(&header_bytes).unwrap();

        anchor.process_new_block(header);

        assert_ne!(anchor.last_tip, BlockHash::all_zeros());
        assert!(anchor.accumulated_work > 0);

        info!("Test Timechain Sync passed.");
    }

    #[test]
    fn test_sanity_anchor() {
        // payload matches the hash of the canonical constitution string:
        // ARKHE_PROTOCOL_OMEGA_215::CONSTITUTION_V1::P1_SOVEREIGNTY::P2_LIFE::P3_TRANSPARENCY::P4_THERMODYNAMICS::P5_CAUSALITY
        let payload = "6a20af369edb96f45c559404a8bb0f631d746ed03a4afa58518fd29dbae73fc00f46";
        let anchor = RealityAnchor::new(payload);

        let state = "ARKHE_PROTOCOL_OMEGA_215::CONSTITUTION_V1::P1_SOVEREIGNTY::P2_LIFE::P3_TRANSPARENCY::P4_THERMODYNAMICS::P5_CAUSALITY";
        assert!(anchor.verify_integrity(state));

        let corrupted = "ARKHE_PROTOCOL_OMEGA_215::P1_SOVEREIGNTY::ALUCINACAO";
        assert!(!anchor.verify_integrity(corrupted));

        info!("Test Sanity Anchor passed.");
    }

    #[test]
    fn test_optimization() {
        let dim = 2;
        let rho = QuantumState::maximally_mixed(dim).density_matrix;
        let mut target = DMatrix::from_element(dim, dim, Complex64::new(0.0, 0.0));
        target[(0,0)] = Complex64::new(1.0, 0.0); // Target pure state |0><0|

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
