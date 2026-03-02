//! Vajra Entropy Monitor - Geometric consensus with attractor morphing

use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use crate::agi_core::{ActivationTensor, Action, Features};
use crate::win_integration::InferenceModel;

#[derive(Debug, Clone, PartialEq)]
pub enum AttractorType {
    TorusKnot { p: u32, q: u32 },
    Halvorsen { a: f64 },
    HalvorsenEmergency,
}

#[derive(Debug, Clone)]
pub struct EntropyMonitor {
    window_size: usize,
    lyapunov_threshold: f64,
    panic_handler: Arc<dyn Fn(&str) + Send + Sync>,

    // Thread-safe state
    state: Arc<RwLock<MonitorState>>,
    particles: Arc<RwLock<Vec<Particle>>>,
    morph_lock: Arc<parking_lot::Mutex<()>>,
    running: Arc<AtomicBool>,
}

#[derive(Debug, Clone)]
struct MonitorState {
    current_attractor: AttractorType,
    lyapunov_history: Vec<f64>,
    coherence_history: Vec<f64>,
    energy_history: Vec<f64>,
    violation_count: u32,
    last_morph: Instant,
    emergency_mode: bool,
}

#[derive(Debug, Clone)]
struct Particle {
    position: [f64; 3],
    velocity: [f64; 3],
}

impl EntropyMonitor {
    pub fn new(
        window_size: usize,
        lyapunov_threshold: f64,
        panic_handler: Box<dyn Fn(&str) + Send + Sync>,
    ) -> Arc<Self> {
        let monitor = Arc::new(Self {
            window_size,
            lyapunov_threshold,
            panic_handler: Arc::from(panic_handler),
            state: Arc::new(RwLock::new(MonitorState {
                current_attractor: AttractorType::TorusKnot { p: 3, q: 5 },
                lyapunov_history: Vec::with_capacity(window_size),
                coherence_history: Vec::with_capacity(window_size),
                energy_history: Vec::with_capacity(window_size),
                violation_count: 0,
                last_morph: Instant::now(),
                emergency_mode: false,
            })),
            particles: Arc::new(RwLock::new(Vec::new())),
            morph_lock: Arc::new(parking_lot::Mutex::new(())),
            running: Arc::new(AtomicBool::new(true)),
        });

        // Start background monitoring thread
        monitor.clone().start_monitoring();

        monitor
    }

    /// Start background monitoring thread
    fn start_monitoring(self: Arc<Self>) {
        std::thread::spawn(move || {
            let monitor_interval = Duration::from_millis(100); // 100ms

            while self.running.load(Ordering::SeqCst) {
                std::thread::sleep(monitor_interval);

                if let Err(e) = self.monitor_cycle() {
                    (self.panic_handler)(&format!("Entropy monitor failed: {}", e));
                    break;
                }
            }
        });
    }

    /// Monitor cycle: check invariants, compute metrics
    fn monitor_cycle(&self) -> Result<(), Box<dyn std::error::Error>> {
        let lyapunov = self.compute_lyapunov_exponent()?;
        let coherence = self.compute_coherence()?;
        let energy = self.compute_energy()?;

        let mut state = self.state.write().unwrap();

        state.lyapunov_history.push(lyapunov);
        state.coherence_history.push(coherence);
        state.energy_history.push(energy);

        if state.lyapunov_history.len() > self.window_size {
            state.lyapunov_history.remove(0);
            state.coherence_history.remove(0);
            state.energy_history.remove(0);
        }

        if lyapunov > self.lyapunov_threshold {
            state.violation_count += 1;

            if state.violation_count > 3 {
                drop(state);
                self.emergency_morph(AttractorType::HalvorsenEmergency);
                return Err("Lyapunov threshold exceeded 3 times".into());
            }
        } else {
            state.violation_count = 0;
        }

        Ok(())
    }

    /// Compute Lyapunov exponent from particle dynamics
    fn compute_lyapunov_exponent(&self) -> Result<f64, Box<dyn std::error::Error>> {
        let particles = self.particles.read().unwrap();

        if particles.len() < 2 {
            return Ok(0.0);
        }

        let mut total_divergence = 0.0;
        let mut count = 0;

        for i in 0..particles.len() {
            for j in (i + 1)..particles.len() {
                let dist = self.distance(&particles[i].position, &particles[j].position);
                if dist > 0.0 {
                    total_divergence += dist.ln();
                }
                count += 1;
            }
        }

        Ok(total_divergence / count as f64)
    }

    /// Compute coherence of the system
    fn compute_coherence(&self) -> Result<f64, Box<dyn std::error::Error>> {
        let particles = self.particles.read().unwrap();

        if particles.is_empty() {
            return Ok(1.0);
        }

        let mut center = [0.0; 3];
        for particle in particles.iter() {
            center[0] += particle.position[0];
            center[1] += particle.position[1];
            center[2] += particle.position[2];
        }

        let n = particles.len() as f64;
        center[0] /= n;
        center[1] /= n;
        center[2] /= n;

        let mut variance = 0.0;
        for particle in particles.iter() {
            let dist = self.distance(&particle.position, &center);
            variance += dist.powi(2);
        }
        variance /= n;

        Ok(1.0 / (1.0 + variance))
    }

    fn compute_energy(&self) -> Result<f64, Box<dyn std::error::Error>> {
        let particles = self.particles.read().unwrap();
        let mut total_energy = 0.0;
        for p in particles.iter() {
            total_energy += p.velocity[0].powi(2) + p.velocity[1].powi(2) + p.velocity[2].powi(2);
        }
        Ok(0.5 * total_energy)
    }

    /// Emergency morph to different attractor
    pub fn emergency_morph(&self, target: AttractorType) {
        let _lock = self.morph_lock.lock();

        {
            let mut state = self.state.write().unwrap();
            if state.emergency_mode && state.current_attractor == target {
                return;
            }
            log::error!("Emergency morph triggered to {:?}", target);
            state.current_attractor = target.clone();
            state.emergency_mode = true;
            state.last_morph = Instant::now();
        }

        self.perform_morph(&target, EasingFunction::Exponential);

        if let AttractorType::HalvorsenEmergency = target {
            (self.panic_handler)("Emergency morph to HalvorsenEmergency");
        }
    }

    /// Perform morph with easing function (Refactored to avoid long-held write locks)
    fn perform_morph(&self, target: &AttractorType, easing: EasingFunction) {
        let steps = 100;

        let original_positions: Vec<[f64; 3]> = {
            let particles = self.particles.read().unwrap();
            particles.iter().map(|p| p.position).collect()
        };

        for step in 0..steps {
            let t = self.easing_function(&easing, step as f64 / steps as f64);

            {
                let mut particles = self.particles.write().unwrap();
                if particles.len() != original_positions.len() { break; } // Safety check

                for (i, particle) in particles.iter_mut().enumerate() {
                    let target_pos = self.compute_target_position(particle, target);

                    particle.position[0] = original_positions[i][0] * (1.0 - t) + target_pos[0] * t;
                    particle.position[1] = original_positions[i][1] * (1.0 - t) + target_pos[1] * t;
                    particle.position[2] = original_positions[i][2] * (1.0 - t) + target_pos[2] * t;
                }
            }

            std::thread::sleep(Duration::from_millis(10));
        }
    }

    pub fn sandboxed_execute<F, T>(&self, f: F) -> Result<T, Box<dyn std::error::Error>>
    where
        F: FnOnce() -> Result<T, Box<dyn std::error::Error>>,
    {
        let start_lyapunov = self.current_lyapunov();
        let result = f()?;
        let end_lyapunov = self.current_lyapunov();
        let delta = (end_lyapunov - start_lyapunov).abs();

        if delta > self.lyapunov_threshold * 0.5 {
            self.flag_coherence_violation();
            return Err("Entropy explosion detected in sandbox".into());
        }

        Ok(result)
    }

    pub fn monitored_forward_pass(
        &self,
        features: &Features,
        model: &Arc<dyn InferenceModel>,
    ) -> Result<ActivationTensor, Box<dyn std::error::Error>> {
        if self.detect_adversarial_pattern(features) {
            return Err("Adversarial pattern detected".into());
        }
        let activation = model.infer(features)?;
        if !self.validate_activation(&activation) {
            return Err("Invalid activation pattern".into());
        }
        Ok(activation)
    }

    pub fn apply_safety_gates(&self, action: Action) -> Action {
        use crate::agi_core::Paradigm;
        match action.paradigm {
            Paradigm::Functional => self.apply_functional_gates(action),
            Paradigm::Imperative => self.apply_imperative_gates(action),
            Paradigm::Agent => self.apply_agent_gates(action),
            Paradigm::Emergency => action,
        }
    }

    pub fn current_lyapunov(&self) -> f64 {
        let state = self.state.read().unwrap();
        state.lyapunov_history.last().copied().unwrap_or(0.0)
    }

    pub fn current_coherence(&self) -> f64 {
        let state = self.state.read().unwrap();
        state.coherence_history.last().copied().unwrap_or(1.0)
    }

    pub fn flag_coherence_violation(&self) {
        let mut state = self.state.write().unwrap();
        state.violation_count += 1;
        if state.violation_count > 5 {
            drop(state);
            self.emergency_morph(AttractorType::HalvorsenEmergency);
        }
    }

    pub fn trigger_panic(&self, reason: &str) {
        (self.panic_handler)(reason);
    }

    pub fn snapshot(&self) -> EntropySnapshot {
        let state = self.state.read().unwrap();
        EntropySnapshot {
            lyapunov: self.current_lyapunov(),
            coherence: self.current_coherence(),
            attractor: state.current_attractor.clone(),
            violation_count: state.violation_count,
            emergency_mode: state.emergency_mode,
        }
    }

    pub fn compute_delta(&self) -> f64 {
        let state = self.state.read().unwrap();
        if state.lyapunov_history.len() < 2 {
            return 0.0;
        }
        let last = state.lyapunov_history.last().unwrap();
        let prev = state.lyapunov_history.get(state.lyapunov_history.len() - 2).unwrap();
        last - prev
    }

    pub fn terminate(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    fn distance(&self, a: &[f64; 3], b: &[f64; 3]) -> f64 {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    }

    fn easing_function(&self, easing: &EasingFunction, t: f64) -> f64 {
        match easing {
            EasingFunction::Linear => t,
            EasingFunction::Cubic => 1.0 - (1.0 - t).powi(3),
            EasingFunction::Exponential => 1.0 - (-5.0 * t).exp(),
            EasingFunction::Logistic => 1.0 / (1.0 + (-8.0 * (t - 0.5)).exp()),
        }
    }

    fn compute_target_position(&self, particle: &Particle, target: &AttractorType) -> [f64; 3] {
        match target {
            AttractorType::TorusKnot { p, q } => {
                let r = 2.0;
                let r_minor = 0.5;
                let theta = (*p as f64) * f64::atan2(particle.position[1], particle.position[0]);
                let phi = (*q as f64) * f64::atan2(
                    particle.position[2],
                    (particle.position[0].powi(2) + particle.position[1].powi(2)).sqrt() - r
                );
                [
                    (r + r_minor * phi.cos()) * theta.cos(),
                    (r + r_minor * phi.cos()) * theta.sin(),
                    r_minor * phi.sin(),
                ]
            }
            AttractorType::Halvorsen { a } => {
                let dt = 0.01;
                let x = particle.position[0];
                let y = particle.position[1];
                let z = particle.position[2];
                let dx = (-a * x - 4.0 * y - 4.0 * z - y.powi(2)) * dt;
                let dy = (-a * y - 4.0 * z - 4.0 * x - z.powi(2)) * dt;
                let dz = (-a * z - 4.0 * x - 4.0 * y - x.powi(2)) * dt;
                [x + dx, y + dy, z + dz]
            }
            AttractorType::HalvorsenEmergency => {
                self.compute_target_position(particle, &AttractorType::Halvorsen { a: 1.89 })
            }
        }
    }

    fn detect_adversarial_pattern(&self, _features: &Features) -> bool {
        false
    }

    fn validate_activation(&self, activation: &ActivationTensor) -> bool {
        activation.data.iter().all(|&x| x.is_finite())
    }

    fn apply_functional_gates(&self, mut action: Action) -> Action {
        action.parameters = self.sanitize_parameters(&action.parameters);
        action
    }

    fn apply_imperative_gates(&self, mut action: Action) -> Action {
        if action.parameters.len() > 1024 * 1024 {
            action.parameters.truncate(1024 * 1024);
        }
        action
    }

    fn apply_agent_gates(&self, action: Action) -> Action {
        action
    }

    fn sanitize_parameters(&self, parameters: &[u8]) -> Vec<u8> {
        parameters.to_vec()
    }
}

#[derive(Debug, Clone)]
pub enum EasingFunction {
    Linear,
    Cubic,
    Exponential,
    Logistic,
}

#[derive(Debug, Clone)]
pub struct EntropySnapshot {
    pub lyapunov: f64,
    pub coherence: f64,
    pub attractor: AttractorType,
    pub violation_count: u32,
    pub emergency_mode: bool,
}
