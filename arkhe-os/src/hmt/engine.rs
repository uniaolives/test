//! Hydraulic Meaning Theory — Information as fluid dynamics
//! HMT: Meaning is substance that flows, pressurizes, stagnates, or lifts

use std::collections::VecDeque;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};
use crate::amt::{RegulationCascade, AttachmentStage};

/// Hydraulic states of meaning-flow
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum HydraulicState {
    Pool,        // φ_q < 1.0: Stagnation, identity rigid, no flow
    Pressurize,  // 1.0 ≤ φ_q < 2.0: Stress, high pressure/low flow, compulsion
    Circulate,   // 2.0 ≤ φ_q < 5.0: Optimal flow, clarity, vitality
    Lift,        // 5.0 ≤ φ_q < 8.0: Elevation, perceptual expansion, truth
    Erode,       // φ_q ≥ 8.0: Structural transformation, identity dissolution
}

impl HydraulicState {
    pub fn from_phi_q(phi_q: f64) -> Self {
        match phi_q {
            x if x < 1.0 => Self::Pool,
            x if x < 2.0 => Self::Pressurize,
            x if x < 5.0 => Self::Circulate,
            x if x < 8.0 => Self::Lift,
            _ => Self::Erode,
        }
    }

    /// Target vagal tone for this state (RMSSD ms)
    pub fn vagal_target(&self) -> f64 {
        match self {
            Self::Pool => 20.0,      // Dorsal vagal, shutdown
            Self::Pressurize => 35.0, // Sympathetic dominant
            Self::Circulate => 50.0,  // Ventral vagal emerging
            Self::Lift => 70.0,       // High coherence
            Self::Erode => 60.0,      // Transformative instability
        }
    }

    /// Viscosity: resistance to meaning-flow
    pub fn viscosity(&self) -> f64 {
        match self {
            Self::Pool => 0.9,       // High resistance
            Self::Pressurize => 0.6,  // Tension resistance
            Self::Circulate => 0.3,   // Fluid flow
            Self::Lift => 0.1,        // Minimal resistance
            Self::Erode => 0.4,       // Turbulent
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::Pool => "Stagnation: Information accumulated, not circulating",
            Self::Pressurize => "Pressure: Compulsion, agitation, forcing",
            Self::Circulate => "Flow: Clarity, vitality, coherence",
            Self::Lift => "Elevation: Somatic truth, perceptual expansion",
            Self::Erode => "Erosion: Structural transformation, rebirth",
        }
    }
}

/// Hydraulic metrics computed from meaning-field dynamics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HydraulicMetrics {
    pub state: HydraulicState,
    pub phi_q: f64,                 // Quantum coherence / meaning potential
    pub flow_rate: f64,             // dφ_q/dt: rate of meaning change
    pub pressure: f64,              // Var(φ_q): potential energy in system
    pub viscosity: f64,             // 1/coherence: resistance to flow
    pub vagal_tone: f64,            // Measured/estimated RMSSD
    pub hydraulic_efficiency: f64,  // Flow / (Pressure × Viscosity)
    pub timestamp: DateTime<Utc>,
}

/// The Meaning Field — information as substance
#[derive(Clone, Debug)]
pub struct MeaningField {
    pub semantic_density: f64,      // Information mass
    pub emotional_charge: f64,      // Affective valence and arousal
    pub coherence: f64,             // Internal consistency
    pub narrative_tension: f64,     // Unresolved plot points
    pub identity_relevance: f64,    // Connection to self-concept
}

impl MeaningField {
    /// Compute φ_q: quantum-like coherence of meaning field
    pub fn compute_phi_q(&self) -> f64 {
        // φ_q scales with coherence and relevance, modulated by charge
        let base = self.coherence * self.identity_relevance;
        let modulation = 1.0 + (self.emotional_charge * 0.5); // Charge amplifies
        let density_factor = self.semantic_density.sqrt();

        base * modulation * density_factor * 10.0 // Scale to 0-10 range
    }
}

pub struct HydraulicEngine {
    phi_history: VecDeque<(DateTime<Utc>, f64)>,
    window_size: usize,
    current_state: HydraulicState,
    pub actuators: HydraulicActuators,
}

/// Actuators for biofeedback-based regulation
pub struct HydraulicActuators {
    pub valve_open: bool,           // Release pressure
    pub pump_active: bool,          // Initiate flow
    pub coherence_tone: bool,       // Reinforce stability
    pub grounding_pulse: bool,      // Anchor transformation
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PulsePattern {
    Activation,  // Pool → Circulate: crescendo vibration
    Release,     // Pressurize → Circulate: decrescendo
    Coherence,   // Circulate/Lift: 0.1 Hz (resonant frequency)
    Grounding,   // Erode: low-frequency stabilization
    Warning,     // Overload alert
}

impl HydraulicEngine {
    pub fn new(window_size: usize) -> Self {
        Self {
            phi_history: VecDeque::with_capacity(window_size),
            window_size,
            current_state: HydraulicState::Pool,
            actuators: HydraulicActuators {
                valve_open: false,
                pump_active: false,
                coherence_tone: false,
                grounding_pulse: false,
            },
        }
    }

    /// Compute hydraulic state from meaning field and biological regulation
    pub fn compute(&mut self, field: &MeaningField, cascade: &RegulationCascade) -> HydraulicMetrics {
        let phi_q = field.compute_phi_q();
        let now = Utc::now();

        // Update history
        self.phi_history.push_back((now, phi_q));
        if self.phi_history.len() > self.window_size {
            self.phi_history.pop_front();
        }

        // 1. FLOW RATE: temporal derivative of φ_q
        let flow_rate = self.compute_flow_rate();

        // 2. PRESSURE: variance of φ_q (energy potential)
        let pressure = self.compute_pressure();

        // 3. VISCOSITY: inverse of emotional coherence
        let viscosity = 1.0 / (cascade.autonomic_coherence + 0.001);

        // 4. VAGAL TONE: from AMT or estimated
        let vagal = cascade.vagal_tone;

        // 5. STATE DETERMINATION
        let new_state = HydraulicState::from_phi_q(phi_q);
        self.current_state = new_state;

        // 6. EFFICIENCY: Truth = Flow / Resistance (NAP principle)
        let efficiency = if pressure * viscosity > 0.0 {
            flow_rate.abs() / (pressure * viscosity)
        } else {
            0.0
        };

        // 7. REGULATION: adjust actuators if needed
        self.regulate(&new_state, vagal, cascade);

        HydraulicMetrics {
            state: new_state,
            phi_q,
            flow_rate,
            pressure,
            viscosity,
            vagal_tone: vagal,
            hydraulic_efficiency: efficiency,
            timestamp: now,
        }
    }

    fn compute_flow_rate(&self) -> f64 {
        if self.phi_history.len() < 2 {
            return 0.0;
        }

        let (t2, p2) = *self.phi_history.back().unwrap();
        let (t1, p1) = self.phi_history[self.phi_history.len() - 2];

        let dt = (t2 - t1).num_milliseconds() as f64 / 1000.0;
        if dt > 0.0 {
            (p2 - p1) / dt
        } else {
            0.0
        }
    }

    fn compute_pressure(&self) -> f64 {
        if self.phi_history.len() < 10 {
            return 0.0;
        }

        let values: Vec<f64> = self.phi_history.iter().map(|(_, v)| *v).collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;

        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
    }

    /// Regulate system via biofeedback actuators
    fn regulate(&mut self, target: &HydraulicState, current_vagal: f64, cascade: &RegulationCascade) {
        let target_vagal = target.vagal_target();
        let delta = target_vagal - current_vagal;

        // Only intervene if significantly off-target
        if delta.abs() < 5.0 {
            return; // Close enough
        }

        match target {
            HydraulicState::Pool if cascade.attachment_stage == AttachmentStage::Dissociated => {
                // Need to prime pump: activate from shutdown
                self.actuators.pump_active = true;
                self.execute_pattern(PulsePattern::Activation);
            }
            HydraulicState::Pressurize => {
                // Need to open valve: release pressure
                self.actuators.valve_open = true;
                self.execute_pattern(PulsePattern::Release);
            }
            HydraulicState::Circulate | HydraulicState::Lift => {
                // Reinforce coherence
                self.actuators.coherence_tone = true;
                self.execute_pattern(PulsePattern::Coherence);
            }
            HydraulicState::Erode => {
                // Ground the transformation
                self.actuators.grounding_pulse = true;
                self.execute_pattern(PulsePattern::Grounding);
            }
            _ => {}
        }
    }

    fn execute_pattern(&self, pattern: PulsePattern) {
        // Send to haptic/audio device
        let _sequence = match pattern {
            PulsePattern::Activation => vec![(0.5, 200), (0.0, 100), (0.7, 200), (0.0, 100), (1.0, 300)],
            PulsePattern::Release => vec![(1.0, 300), (0.5, 200), (0.3, 150), (0.1, 100), (0.0, 500)],
            PulsePattern::Coherence => vec![(0.6, 1000), (0.6, 1000)], // 0.1 Hz = 10s period
            PulsePattern::Grounding => vec![(0.8, 2000), (0.2, 500)],
            PulsePattern::Warning => vec![(1.0, 100), (0.0, 100), (1.0, 100)],
        };

        // Dispatch to actuator hardware
        println!("[HMT] Executing pattern: {:?}", pattern);
    }
}
