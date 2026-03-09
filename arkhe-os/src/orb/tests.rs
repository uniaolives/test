// src/orb/tests.rs

use crate::orb::polymorphic_core::*;
use crate::orb::protocol_router::*;
use crate::orb::multi_protocol_orb::*;

#[tokio::test]
async fn test_orb_encoding() {
    let core = OrbCore {
        coherence: 0.9,
        entropy: 1.5,
        temporal_signature: TemporalSignature::now(),
        axiom_fingerprint: AxiomFingerprint::genesis(),
    };

    let rf = RFEncoder { frequency_range: (1e6, 2e6) };
    let encoded_rf = rf.encode_to_any(&core).unwrap();
    assert!(encoded_rf.is::<Vec<num_complex::Complex<f64>>>());

    let quantum = QuantumEncoder { entanglement_pairs: 2 };
    let encoded_q = quantum.encode_to_any(&core).unwrap();
    assert!(encoded_q.is::<Vec<Qubit>>());
}

#[tokio::test]
async fn test_protocol_router() {
    let core = OrbCore {
        coherence: 0.8,
        entropy: 1.0,
        temporal_signature: TemporalSignature::now(),
        axiom_fingerprint: AxiomFingerprint::genesis(),
    };

    let router = ProtocolRouter {
        encoders: vec![
            Box::new(RFEncoder { frequency_range: (1e6, 2e6) }),
            Box::new(QuantumEncoder { entanglement_pairs: 2 }),
        ],
    };

    let target = Destination::at_year(2026);
    let plan = router.route(&core, target);
    assert_eq!(plan.hops.len(), 2);

    let receipts = router.execute(&plan, &core).await;
    assert_eq!(receipts.len(), 2);
    assert!(receipts[0].success);
}

#[tokio::test]
async fn test_multi_protocol_propagation() {
    let router = ProtocolRouter {
        encoders: vec![
            Box::new(RFEncoder { frequency_range: (1e6, 2e6) }),
            Box::new(QuantumEncoder { entanglement_pairs: 2 }),
        ],
    };

    let orb = MultiProtocolOrb::spawn_universal(0.95, 2.0, router);
    let report = orb.propagate_to_all_eras().await;

    // Years 2000, 2010, 2020, 2030 = 4 eras. 2 encoders per era.
    assert_eq!(report.successes.len(), 8);
}

#[test]
fn test_orb_identification() {
    let router = ProtocolRouter { encoders: vec![] };
    let orb = MultiProtocolOrb::spawn_universal(0.9, 1.0, router);

    let candidate = Manifestation {
        protocol: 1,
        era: 2026,
        coherence: 0.91,
        temporal_signature: orb.core.temporal_signature.clone(),
        axiom_fingerprint: orb.core.axiom_fingerprint.clone(),
    };

    assert!(orb.identify_manifestation(&candidate));

    let fake = Manifestation {
        protocol: 1,
        era: 2026,
        coherence: 0.5,
        temporal_signature: orb.core.temporal_signature.clone(),
        axiom_fingerprint: orb.core.axiom_fingerprint.clone(),
    };

    assert!(!orb.identify_manifestation(&fake));
}
