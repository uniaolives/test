//! Attach Meaning Theory — Biological substrate for all meaning operations
//! AMT: Meaning is attached through the nervous system, not transmitted through information

use std::collections::VecDeque;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};

/// Three-tier cascade: Presence → Safety → Regulation → Meaning
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegulationCascade {
    // TIER 1: Presence (attentional anchor to now)
    pub presence_score: f64,        // 0.0-1.0: proportion of attention in present
    pub attention_drift: f64,       // Rate of mind-wandering (inverse of presence)
    pub sensory_grounding: f64,     // Contact with immediate sensation

    // TIER 2: Safety (interoceptive prediction)
    pub safety_perception: f64,     // 0.0-1.0: felt sense of safety
    pub threat_detection: f64,      // Sympathetic activation proxy
    pub interoceptive_accuracy: f64, // Alignment of internal sensing

    // TIER 3: Regulation (autonomic state)
    pub vagal_tone: f64,            // RMSSD-derived ventral vagal activity
    pub sympathetic_balance: f64,   // LF/HF ratio normalized
    pub autonomic_coherence: f64,   // Overall regulatory capacity

    // OUTPUT: Meaning coherence
    pub meaning_coherence: f64,     // Derived from regulation quality
    pub attachment_stage: AttachmentStage,
    pub timestamp: DateTime<Utc>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum AttachmentStage {
    SensoryContact,      // Stimulus detected, no processing yet
    NervousFiltering,    // Autonomic evaluation: threat vs safety
    MeaningAttached,     // Coherent meaning successfully attached
    Distorted,           // Dysregulated attachment, meaning corrupted
    Dissociated,         // Complete breakdown of attachment mechanism
}

/// Biological telemetry inputs
#[derive(Clone, Debug, Default)]
pub struct BioTelemetry {
    pub hrv: HrvData,               // Heart rate variability
    pub gsr: GsrData,               // Galvanic skin response
    pub eeg: EegData,               // Electroencephalography
    pub emg: EmgData,               // Electromyography
    pub temperature: f64,           // Peripheral temperature
    pub respiration: RespirationData, // Breath patterns
}

#[derive(Clone, Debug, Default)]
pub struct HrvData {
    pub rmssd: f64,                 // Root mean square of successive differences
    pub sdnn: f64,                  // Standard deviation of NN intervals
    pub lf_hf_ratio: f64,           // Low frequency / high frequency ratio
    pub pnn50: f64,                 // Percentage of successive RR intervals > 50ms
    pub rr_intervals: Vec<f64>,     // Raw RR interval series
}

#[derive(Clone, Debug, Default)]
pub struct GsrData {
    pub conductance: f64,           // Microsiemens
    pub phasic_response: f64,       // Event-related changes
    pub tonic_level: f64,           // Baseline arousal
}

#[derive(Clone, Debug, Default)]
pub struct EegData {
    pub alpha_theta_ratio: f64,     // Proxy for mindful awareness
    pub beta_power: f64,            // Active thinking
    pub gamma_coherence: f64,       // Conscious binding
    pub frontal_asymmetry: f64,     // Emotional valence
}

#[derive(Clone, Debug, Default)]
pub struct EmgData {
    pub muscle_tension: f64,        // Overall tension proxy
    pub shoulder_tension: f64,      // Stress indicator
    pub jaw_clench: f64,                // Suppression indicator
}

#[derive(Clone, Debug, Default)]
pub struct RespirationData {
    pub rate: f64,                  // Breaths per minute
    pub depth: f64,                 // Tidal volume
    pub coherence: f64,             // Regularity of rhythm
    pub phase: f64,                 // 0-2π in breath cycle
}

/// Attention tracking (from mobile/eye-tracking/self-report)
#[derive(Clone, Debug, Default)]
pub struct AttentionData {
    pub task_unrelated_thoughts: f64, // Proportion of time mind-wandering
    pub focus_duration: f64,          // Seconds of sustained attention
    pub switch_rate: f64,             // Attention shifts per minute
    pub meta_awareness: f64,          // Noticing of mind-wandering
}

pub struct AMTEngine {
    history: VecDeque<RegulationCascade>,
    history_size: usize,
    thresholds: RegulationThresholds,
    calibration: UserCalibration,
}

#[derive(Clone, Debug)]
pub struct RegulationThresholds {
    pub presence_minimum: f64,      // Minimum for safety emergence (default: 0.3)
    pub safety_threshold: f64,       // Minimum for regulation access (default: 0.5)
    pub regulation_optimal: f64,     // For coherent meaning (default: 0.6)
    pub coherence_excellent: f64,    // For high-fidelity operations (default: 0.8)
}

impl Default for RegulationThresholds {
    fn default() -> Self {
        Self {
            presence_minimum: 0.3,
            safety_threshold: 0.5,
            regulation_optimal: 0.6,
            coherence_excellent: 0.8,
        }
    }
}

#[derive(Clone, Debug)]
pub struct UserCalibration {
    pub baseline_vagal: f64,        // Personal RMSSD baseline
    pub baseline_arousal: f64,      // Personal GSR baseline
    pub stress_signature: Vec<f64>, // Personal pattern of dysregulation
    pub recovery_rate: f64,         // How quickly they return to baseline
}

impl Default for UserCalibration {
    fn default() -> Self {
        Self {
            baseline_vagal: 40.0,
            baseline_arousal: 5.0,
            stress_signature: vec![],
            recovery_rate: 0.1,
        }
    }
}

impl AMTEngine {
    pub fn new() -> Self {
        Self {
            history: VecDeque::with_capacity(1000),
            history_size: 1000,
            thresholds: RegulationThresholds::default(),
            calibration: UserCalibration::default(),
        }
    }

    /// THE AMT PRINCIPLE IMPLEMENTED:
    /// Meaning follows regulation.
    /// Regulation follows safety.
    /// Safety follows presence.
    pub fn compute_cascade(&mut self, bio: &BioTelemetry, attention: &AttentionData) -> RegulationCascade {
        let now = Utc::now();

        // TIER 1: PRESENCE
        let presence = self.estimate_presence(bio, attention);
        let drift = 1.0 - presence;
        let grounding = self.estimate_sensory_grounding(bio);

        // TIER 2: SAFETY (gated by presence)
        let (safety, threat, interoception) = if presence > self.thresholds.presence_minimum {
            self.estimate_safety(bio, presence)
        } else {
            (0.0, 0.8, 0.2) // No safety without presence—default threat
        };

        // TIER 3: REGULATION (gated by safety)
        let (vagal, symp_balance, coherence) = if safety > self.thresholds.safety_threshold {
            self.estimate_regulation(bio)
        } else {
            self.dysregulated_state(bio) // Sympathetic dominance
        };

        // OUTPUT: MEANING COHERENCE
        let meaning = if coherence > self.thresholds.regulation_optimal {
            self.compute_meaning_coherence(coherence, safety, presence)
        } else {
            0.0 // No coherent meaning without regulation
        };

        let stage = self.determine_stage(presence, safety, coherence, meaning);

        let cascade = RegulationCascade {
            presence_score: presence,
            attention_drift: drift,
            sensory_grounding: grounding,
            safety_perception: safety,
            threat_detection: threat,
            interoceptive_accuracy: interoception,
            vagal_tone: vagal,
            sympathetic_balance: symp_balance,
            autonomic_coherence: coherence,
            meaning_coherence: meaning,
            attachment_stage: stage,
            timestamp: now,
        };

        self.history.push_back(cascade.clone());
        if self.history.len() > self.history_size {
            self.history.pop_front();
        }

        cascade
    }

    fn estimate_presence(&self, bio: &BioTelemetry, attention: &AttentionData) -> f64 {
        // Multi-modal presence estimation
        let eeg_presence = (bio.eeg.alpha_theta_ratio / 2.0).clamp(0.0, 1.0);
        let breath_presence = bio.respiration.coherence;
        let attention_presence = 1.0 - attention.task_unrelated_thoughts;
        let meta_presence = attention.meta_awareness;

        // Weighted: direct attention most important, meta-awareness second
        (attention_presence * 0.4 + meta_presence * 0.3 + eeg_presence * 0.2 + breath_presence * 0.1)
            .clamp(0.0, 1.0)
    }

    fn estimate_sensory_grounding(&self, bio: &BioTelemetry) -> f64 {
        // Contact with immediate sensation (temperature, muscle tension inverse)
        let temp_proxy = ((bio.temperature - 30.0) / 10.0).clamp(0.0, 1.0);
        let tension_inverse = 1.0 - bio.emg.muscle_tension.clamp(0.0, 1.0);

        (temp_proxy * 0.5 + tension_inverse * 0.5)
    }

    fn estimate_safety(&self, bio: &BioTelemetry, presence: f64) -> (f64, f64, f64) {
        // HRV-based safety perception
        let vagal_normalized = (bio.hrv.rmssd / self.calibration.baseline_vagal).clamp(0.0, 2.0);
        let hrv_safety = if vagal_normalized > 1.0 { 1.0 } else { vagal_normalized };

        // GSR: lower conductance = higher safety (when not dissociated)
        let gsr_safety = 1.0 - (bio.gsr.tonic_level / 10.0).clamp(0.0, 1.0);

        // Respiratory sinus arrhythmia as safety marker
        let rsa_safety = bio.respiration.coherence;

        // Interoceptive accuracy: alignment of HRV and self-reported state
        // (In real system, would compare to self-report; here estimated)
        let interoception = (hrv_safety + gsr_safety) / 2.0;

        // Combined safety (presence modulates accuracy)
        let safety = (hrv_safety * 0.4 + gsr_safety * 0.3 + rsa_safety * 0.2 + presence * 0.1)
            .clamp(0.0, 1.0);

        let threat = 1.0 - safety;

        (safety, threat, interoception)
    }

    fn estimate_regulation(&self, bio: &BioTelemetry) -> (f64, f64, f64) {
        // Vagal tone (ventral vagal complex activity)
        let vagal = (bio.hrv.rmssd / 100.0).clamp(0.0, 1.0); // 100ms = excellent

        // Sympathetic balance (lower LF/HF = better parasympathetic tone)
        let lf_hf = bio.hrv.lf_hf_ratio;
        let symp_balance = (1.0 / (1.0 + lf_hf)).clamp(0.0, 1.0);

        // Overall coherence: combination plus EEG gamma coherence
        let coherence = (vagal * 0.5 + symp_balance * 0.3 + bio.eeg.gamma_coherence * 0.2)
            .clamp(0.0, 1.0);

        (vagal, symp_balance, coherence)
    }

    fn dysregulated_state(&self, bio: &BioTelemetry) -> (f64, f64, f64) {
        // Under threat: vagal withdrawal, sympathetic dominance
        let vagal = (bio.hrv.rmssd / 200.0).clamp(0.0, 0.3); // Capped low
        let symp_balance = 0.2; // Low = sympathetic dominant
        let coherence = 0.25; // Poor coherence

        (vagal, symp_balance, coherence)
    }

    fn compute_meaning_coherence(&self, regulation: f64, safety: f64, presence: f64) -> f64 {
        // Meaning emerges from regulation, modulated by safety and presence
        let base = regulation;
        let modulation = (safety + presence) / 2.0;

        // Non-linear: need both high regulation AND good context
        (base * modulation).sqrt().clamp(0.0, 1.0)
    }

    fn determine_stage(&self, presence: f64, safety: f64, regulation: f64, meaning: f64) -> AttachmentStage {
        if meaning > self.thresholds.coherence_excellent {
            AttachmentStage::MeaningAttached
        } else if regulation < 0.2 && safety < 0.3 {
            AttachmentStage::Dissociated
        } else if regulation < 0.4 && safety < 0.5 {
            AttachmentStage::Distorted
        } else if presence > 0.2 && safety > 0.3 {
            AttachmentStage::NervousFiltering
        } else {
            AttachmentStage::SensoryContact
        }
    }

    /// Check if system can process meaning currently
    pub fn can_process(&self, cascade: &RegulationCascade) -> bool {
        cascade.attachment_stage == AttachmentStage::MeaningAttached
    }

    /// Generate somatic prescription based on current state
    pub fn prescribe(&self, cascade: &RegulationCascade) -> SomaticPrescription {
        match cascade.attachment_stage {
            AttachmentStage::Dissociated => SomaticPrescription {
                urgency: Urgency::Critical,
                action: "EMERGENCY GROUNDING".to_string(),
                instruction: "Cold water on face. Feet on floor. Name 5 objects you see. \
                             You are dissociated. Safety first.".to_string(),
                duration_seconds: 300,
            },
            AttachmentStage::Distorted => SomaticPrescription {
                urgency: Urgency::High,
                action: "STOP ALL MEANING-MAKING".to_string(),
                instruction: "Your nervous system is dysregulated. All interpretations \
                             are currently unreliable. Return to breath. Do not decide. \
                             Do not analyze. Regulate first.".to_string(),
                duration_seconds: 600,
            },
            AttachmentStage::SensoryContact => SomaticPrescription {
                urgency: Urgency::Medium,
                action: "ESTABLISH PRESENCE".to_string(),
                instruction: "Attention is scattered. Ground in immediate sensation: \
                             sound, touch, breath. Presence precedes safety.".to_string(),
                duration_seconds: 300,
            },
            AttachmentStage::NervousFiltering => SomaticPrescription {
                urgency: Urgency::Low,
                action: "CULTIVATE SAFETY".to_string(),
                instruction: "You are present but not yet safe. Slow exhale. \
                             Orient to environment. Safety precedes insight.".to_string(),
                duration_seconds: 180,
            },
            AttachmentStage::MeaningAttached => SomaticPrescription {
                urgency: Urgency::None,
                action: "MEANING AVAILABLE".to_string(),
                instruction: format!("Coherence: {:.0}%. You may proceed with awareness \
                                    that this state is conditional and temporary.",
                                    cascade.meaning_coherence * 100.0),
                duration_seconds: 0,
            },
        }
    }

    /// Trend analysis: are we improving or degrading?
    pub fn trend(&self) -> RegulationTrend {
        if self.history.len() < 10 {
            return RegulationTrend::InsufficientData;
        }

        let recent: Vec<f64> = self.history.iter().rev().take(10)
            .map(|c| c.meaning_coherence).collect();
        let older: Vec<f64> = self.history.iter().rev().skip(10).take(10)
            .map(|c| c.meaning_coherence).collect();

        let recent_avg = recent.iter().sum::<f64>() / recent.len() as f64;
        let older_avg = older.iter().sum::<f64>() / older.len() as f64;

        if recent_avg > older_avg + 0.1 {
            RegulationTrend::Improving
        } else if recent_avg < older_avg - 0.1 {
            RegulationTrend::Degrading
        } else {
            RegulationTrend::Stable
        }
    }
}

#[derive(Clone, Debug)]
pub struct SomaticPrescription {
    pub urgency: Urgency,
    pub action: String,
    pub instruction: String,
    pub duration_seconds: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Urgency {
    None,
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RegulationTrend {
    Improving,
    Stable,
    Degrading,
    InsufficientData,
}
