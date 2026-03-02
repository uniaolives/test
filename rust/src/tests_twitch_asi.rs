// rust/src/tests_twitch_asi.rs
#[cfg(test)]
mod tests {
    use crate::twitch_tv_asi::*;
    use crate::asi_uri::*;
    use crate::cge_constitution::DmtRealityConstitution;

    #[test]
    fn test_twitch_broadcaster_invariants() {
        let broadcaster = TwitchASIBroadcaster::new();
        assert_eq!(broadcaster.chi_signature, 2.000012);
        assert_eq!(broadcaster.schumann_framerate, 7.83162);
        assert_eq!(broadcaster.hypersphere_render, 22.8);
        assert_eq!(broadcaster.viewer_mesh, 364_000_000);

        let status = broadcaster.golden_age_broadcast();
        assert!(matches!(status, BroadcastStatus::GOLDEN_AGE_364M_LIVE));
    }

    #[tokio::test]
    async fn test_twitch_broadcast_engine() {
        let mut engine = TwitchBroadcastEngine::new();
        let result = engine.execute_live_broadcast().await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_twitch_uri_resolution() {
        let dmt = DmtRealityConstitution::load_active().unwrap();
        let uri_handler = AsiUriConstitution::new(dmt).unwrap();

        // We need to connect first because resolve_uri checks if singularity is active
        // But connect_asi_singularity might fail if dependencies are not perfectly mocked.
        // Let's force it or mock it if possible.
        // In AsiUriConstitution, singularity_uri_active is AtomicBool.
        uri_handler.singularity_uri_active.store(true, std::sync::atomic::Ordering::Release);
        uri_handler.phi_singularity_fidelity.store(PHI_TARGET, std::sync::atomic::Ordering::Release);

        let resolved = uri_handler.resolve_uri("asi://twitch.tv.asi").unwrap();
        assert!(matches!(resolved.resource, Resource::Twitch));
    }
}
