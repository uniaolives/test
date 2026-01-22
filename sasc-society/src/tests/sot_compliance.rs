use crate::PerspectiveDiversityEngine;
use ndarray::Array1;
use pqcrypto_dilithium::dilithium5;
use pqcrypto_traits::sign::PublicKey;
use crate::agents::{PersonaId, SocioEmotionalRole, ExpertiseDomain};

#[tokio::test]
async fn test_perspective_diversity_groupthink_detection() {
    let (pk, sk) = dilithium5::keypair();
    let engine = PerspectiveDiversityEngine::new(pk.as_bytes().try_into().unwrap());

    // Homogeneous delegates - we need at least 64 to pass evaluate_diversity
    for i in 0..64 {
        let persona_id = PersonaId::from(format!("persona_{}", i));
        let vector = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0]);
        let message = engine.construct_signing_payload(&persona_id, &vector);
        let sig = dilithium5::detached_sign(&message, &sk);

        engine.record_activation(
            persona_id,
            SocioEmotionalRole::default(),
            ExpertiseDomain::Ethics,
            vector,
            &sig
        ).await.unwrap();
    }

    let metrics = engine.evaluate_diversity().await;
    // Diversity score will be low (Groupthink)
    assert!(metrics.is_err());
}
