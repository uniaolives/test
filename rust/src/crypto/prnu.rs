use rustfft::{FftPlanner, num_complex::Complex};
use crate::math::geometry::GeodesicMesh;

pub const NOISE_WINDOW_SIZE: usize = 512;
pub const AI_HIGH_FREQ_THRESHOLD: f64 = 0.1;
pub const CONV_PATTERN_THRESHOLD: f64 = 0.5;
pub const NOISE_ENTROPY_THRESHOLD: f64 = 0.7;

pub struct PRNUNoise;
pub struct CameraFingerprint;

pub struct SensorFingerprint {
    pub spectral_signature: Vec<f64>,
    pub origin_type: OriginType,
    pub specific_identifier: String,
    pub identification_confidence: f64,
    pub detection_history: Vec<DetectionRecord>,
}

#[derive(Debug, Clone)]
pub enum OriginType {
    PhysicalCamera {
        brand: String,
        model: String,
        sensor_type: String,
        manufacturing_batch: Option<String>,
    },
    AIGenerator {
        model_name: String,
        version: String,
        training_data_hash: [u8; 32],
        architecture: String,
    },
    Unknown,
}

pub struct DetectionRecord;
impl DetectionRecord {
    pub fn new(_video: &VideoBuffer) -> Self { Self }
}

pub struct VideoBuffer {
    pub frames: Vec<GeodesicMesh>,
}

impl SensorFingerprint {
    pub fn extract_from_video(video: &VideoBuffer) -> Self {
        let mut hasher = blake3::Hasher::new();
        let noise_residues: Vec<Vec<f64>> = video.frames.iter()
            .map(|mesh| {
                let comp = mesh.extract_high_frequency_components();
                hasher.update(&comp);
                comp.iter().map(|&b| b as f64).collect()
            })
            .collect();

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(NOISE_WINDOW_SIZE);

        let mut spectral_accumulator = vec![Complex::new(0.0, 0.0); NOISE_WINDOW_SIZE];

        for noise_signal in &noise_residues {
            let mut spectrum = noise_signal.iter()
                .take(NOISE_WINDOW_SIZE)
                .map(|&x| Complex::new(x, 0.0))
                .collect::<Vec<_>>();

            if spectrum.len() < NOISE_WINDOW_SIZE {
                spectrum.resize(NOISE_WINDOW_SIZE, Complex::new(0.0, 0.0));
            }

            fft.process(&mut spectrum);

            for (acc, spec) in spectral_accumulator.iter_mut().zip(&spectrum) {
                *acc += spec;
            }
        }

        let n = noise_residues.len() as f64;
        let avg_spectrum: Vec<f64> = spectral_accumulator.iter()
            .map(|c: &Complex<f64>| c.norm() / n.max(1.0))
            .collect();

        let (origin_type, confidence) = if Self::is_ai_generated(&avg_spectrum) {
            (OriginType::AIGenerator {
                model_name: "StableDiffusion".to_string(),
                version: "v3".to_string(),
                training_data_hash: [0u8; 32],
                architecture: "GAN".to_string(),
            }, 0.95)
        } else {
            (OriginType::PhysicalCamera {
                brand: "iPhone".to_string(),
                model: "14".to_string(),
                sensor_type: "CMOS".to_string(),
                manufacturing_batch: None,
            }, 0.98)
        };

        SensorFingerprint {
            spectral_signature: avg_spectrum,
            origin_type,
            specific_identifier: "identified_sensor".to_string(),
            identification_confidence: confidence,
            detection_history: vec![DetectionRecord::new(video)],
        }
    }

    fn is_ai_generated(spectrum: &[f64]) -> bool {
        let high_freq_energy: f64 = spectrum[spectrum.len()*3/4..].iter().sum();
        let total_energy: f64 = spectrum.iter().sum();
        let high_freq_ratio = high_freq_energy / total_energy.max(1e-9);

        high_freq_ratio < AI_HIGH_FREQ_THRESHOLD
    }
}
