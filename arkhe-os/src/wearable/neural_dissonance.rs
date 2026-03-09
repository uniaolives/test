// src/wearable/neural_dissonance.rs

/// Representa o sinal neural capturado via wearable (EEG/Neuralink)
pub struct NeuralSignal {
    pub error_rate: f64,
}

impl NeuralSignal {
    pub fn prediction_error(&self) -> f64 {
        self.error_rate
    }
}

pub struct NeuralDissonanceDetector {
    pub baseline_surprise: f64,
}

impl NeuralDissonanceDetector {
    pub fn new(baseline: f64) -> Self {
        Self { baseline_surprise: baseline }
    }

    /// Verifica se o cérebro está rejeitando a realidade visual
    pub fn check_brain_rejection(&self, neural_stream: &NeuralSignal) -> bool {
        let current_surprise = neural_stream.prediction_error();

        // Se o erro de predição for muito alto, o conteúdo é "irreal" (Uncanny Valley)
        current_surprise > self.baseline_surprise * 3.0
    }
}
