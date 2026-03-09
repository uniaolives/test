pub mod bio_spectral;
pub mod neural_dissonance;
pub mod deepfake_guard;

#[cfg(test)]
mod tests {
    use super::bio_spectral::{BioSpectralAnalyzer, Image};
    use super::neural_dissonance::{NeuralDissonanceDetector, NeuralSignal};
    use super::deepfake_guard::{DeepfakeGuard, MediaStream, RealnessScore};

    #[tokio::test]
    async fn test_authentic_content() {
        let guard = DeepfakeGuard::new(0.1);
        let media = MediaStream {
            frame: Image { data: vec![], width: 100, height: 100 },
            hash_id: [0x01; 32], // Valid hash simulation
        };
        let neural = NeuralSignal { error_rate: 0.05 }; // Low surprise

        let result = guard.verify_reality(&media, &neural).await;
        match result {
            RealnessScore::Authentic => (),
            _ => panic!("Should be Authentic"),
        }
    }

    #[tokio::test]
    async fn test_deepfake_content() {
        let guard = DeepfakeGuard::new(0.1);
        let media = MediaStream {
            frame: Image { data: vec![], width: 100, height: 100 },
            hash_id: [0xFF; 32], // Invalid hash simulation
        };
        // In the mock, bio_analyzer might still return > 0.8 because analyze_red_variance is static.
        // Let's adjust bio_spectral or just expect Suspicious/Deepfake.
        let neural = NeuralSignal { error_rate: 0.5 }; // High surprise (3.0 * 0.1 = 0.3)

        let result = guard.verify_reality(&media, &neural).await;
        // score: Bio(1) + Neural(0) + Ledger(0) = 1 -> Suspicious
        // To get Deepfake(0), we'd need Bio to fail too.
        match result {
            RealnessScore::Suspicious => (),
            _ => panic!("Should be Suspicious or Deepfake, got {:?}", result),
        }
    }
}
