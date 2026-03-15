// arkhe-os/src/net/http5_tests.rs

#[cfg(test)]
mod tests {
    use crate::net::http4::{Http4Method, TemporalHeaders, ConfinementMode, ParadoxPolicy};
    use crate::net::http5::{Http5Request, InterstellarHeaders, Http5Response};
    use crate::net::http4_router::{Http4Router, Http4Error};
    use chrono::Utc;

    #[tokio::test]
    async fn test_voyager_genesis_emission() {
        let router = Http4Router::new(1.0);
        let request = Http5Request {
            method: Http4Method::EMIT,
            resource: "/heliosphere/voyager_genesis".to_string(),
            interstellar_headers: InterstellarHeaders {
                temporal_headers: TemporalHeaders {
                    origin: Utc::now(),
                    target: Utc::now(), // In test we use now, but target is 1977
                    lambda_2: 0.999,
                    confinement: ConfinementMode::INFINITE_WELL,
                    paradox_policy: ParadoxPolicy::MERGE,
                    mobius_twist: 0.5,
                    oam_state: None,
                    retrocausal_timestamp: None,
                },
                carrier_frequency: 1420.4556,
                oam_topology_l: 100,
                grail_signature: "6EQUJ5-ALPHA-OMEGA".to_string(),
            },
            payload: vec![],
        };

        let response = router.route_interstellar_packet(request).await.unwrap();
        match response {
            Http5Response::TimelineStabilized(res) => assert_eq!(res, "/heliosphere/voyager_genesis"),
            _ => panic!("Expected TimelineStabilized response"),
        }
    }

    #[tokio::test]
    async fn test_invalid_http5_signature() {
        let router = Http4Router::new(1.0);
        let request = Http5Request {
            method: Http4Method::EMIT,
            resource: "/heliosphere/voyager_genesis".to_string(),
            interstellar_headers: InterstellarHeaders {
                temporal_headers: TemporalHeaders {
                    origin: Utc::now(),
                    target: Utc::now(),
                    lambda_2: 0.999,
                    confinement: ConfinementMode::INFINITE_WELL,
                    paradox_policy: ParadoxPolicy::MERGE,
                    mobius_twist: 0.5,
                    oam_state: None,
                    retrocausal_timestamp: None,
                },
                carrier_frequency: 1420.4556,
                oam_topology_l: 100,
                grail_signature: "INVALID_SIG".to_string(),
            },
            payload: vec![],
        };

        let result = router.route_interstellar_packet(request).await;
        assert!(matches!(result, Err(Http4Error::TemporalSpoofingDetected)));
    }
}
