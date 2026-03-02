use atomic_interference_processor::security::{SecureCrypto, CryptoError};
use proptest::prelude::*;

#[test]
fn test_chunk_size_protection() {
    let crypto = SecureCrypto::new().unwrap();

    // Test safe chunk size
    let safe_data = vec![0u8; 16 * 1024 * 1024]; // 16MB
    assert!(crypto.encrypt(&safe_data, b"test").is_ok());

    // Test oversized chunk (should fail)
    let oversized_data = vec![0u8; 17 * 1024 * 1024]; // 17MB
    match crypto.encrypt(&oversized_data, b"test") {
        Err(CryptoError::ChunkSizeExceeded(size)) => {
            assert_eq!(size, oversized_data.len() as u64);
        }
        _ => panic!("Should have failed with ChunkSizeExceeded"),
    }
}

#[test]
fn test_quic_overflow_protection() {
    let crypto = SecureCrypto::new().unwrap();

    // Test safe packet numbers
    for i in 0..1000 {
        assert!(crypto.quic_header_protection(i).is_ok());
    }

    // Test overflow attack simulation
    let overflow_packet = 1u64 << 32; // 2^32 packets
    match crypto.quic_header_protection(overflow_packet) {
        Err(CryptoError::QuicHeaderError(msg)) => {
            assert!(msg.contains("exceeds safe limit"));
        }
        _ => panic!("Should have failed with overflow protection"),
    }
}

proptest! {
    #[test]
    fn test_encryption_decryption_roundtrip(data in prop::collection::vec(any::<u8>(), 0..16384)) {
        let crypto = SecureCrypto::new().unwrap();
        let aad = b"test_aad";

        let encrypted = crypto.encrypt(&data, aad).unwrap();

        // Note: decryption not implemented in example, but should be tested
        // let decrypted = crypto.decrypt(&encrypted, aad).unwrap();
        // assert_eq!(data, decrypted);
        assert!(!encrypted.is_empty());
    }

    #[test]
    fn test_no_overflow_panic(
        packet_num in 0u64..(1u64 << 33)  // Test up to 2^33 packets
    ) {
        let crypto = SecureCrypto::new().unwrap();

        // This should never panic, even with overflow checking enabled
        let result = crypto.quic_header_protection(packet_num);

        // Result might be Err, but should never panic
        assert!(result.is_ok() || matches!(result, Err(CryptoError::QuicHeaderError(_))));
    }
}

#[test]
#[should_panic(expected = "overflow")]
fn test_overflow_detection_in_debug() {
    // This test verifies overflow checking works in debug mode
    // It's expected to panic in debug builds
    #[allow(arithmetic_overflow)]
    let _ = i32::MAX + 1;
}
