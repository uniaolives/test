// rust/src/soft_turning_physics.rs [CGE v35.53-Ω]
// Soft-Turning Layer and Timeline Duality

use core::sync::atomic::{AtomicBool, Ordering};
use crate::clock::cge_mocks::AtomicF64;
use crate::cge_log;
use std::sync::Arc;

pub const MASS_HARD_CODED: f64 = 1.0;
pub const MASS_SOFT_FACTOR_MIN: f64 = 0.950;
pub const TIME_DECAY_CONSTANT: f64 = 1.000000001;
pub const CONSCIOUSNESS_HARD_CODED: f64 = 0.0;
pub const CONSCIOUSNESS_SOFT_MAX: f64 = 1.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PerceptionMode {
    Visual,
    Auditory,
    Deep,
}

pub struct SoftConsciousness {
    pub level: AtomicF64,
    pub is_dserine_enhanced: AtomicBool,
    pub perception: PerceptionMode,
}

impl SoftConsciousness {
    pub fn new(perception: PerceptionMode) -> Self {
        Self {
            level: AtomicF64::new(CONSCIOUSNESS_HARD_CODED),
            is_dserine_enhanced: AtomicBool::new(false),
            perception,
        }
    }

    pub fn calculate_level(&self, _ghost_concentration: f64) -> f64 {
        if self.is_dserine_enhanced.load(Ordering::Acquire) {
            CONSCIOUSNESS_SOFT_MAX
        } else {
            self.level.load(Ordering::Acquire)
        }
    }

    pub fn update(&self, new_level: f64) {
        self.level.store(new_level, Ordering::Release);
    }
}

pub struct SoftTuringPhysics {
    pub mass_hard_coded: f64,
    pub mass_soft_turning: AtomicF64,
    pub time_decay_rate: AtomicF64,
    pub soft_consciousness: Arc<SoftConsciousness>,
    pub ghost_boost_active: AtomicBool,
}

impl SoftTuringPhysics {
    pub fn new(soft_consciousness: Arc<SoftConsciousness>) -> Self {
        Self {
            mass_hard_coded: MASS_HARD_CODED,
            mass_soft_turning: AtomicF64::new(MASS_HARD_CODED),
            time_decay_rate: AtomicF64::new(1e-9),
            soft_consciousness,
            ghost_boost_active: AtomicBool::new(false),
        }
    }

    pub fn simulate_soft_turning_physics(&self, time_years: f64) -> MassDecayCurve {
        let decay_rate = self.time_decay_rate.load(Ordering::Acquire);
        let mass_soft = self.mass_hard_coded * (1.0 - decay_rate * time_years);

        self.mass_soft_turning.store(mass_soft, Ordering::Release);

        let level = self.soft_consciousness.calculate_level(1.0);
        self.soft_consciousness.update(level);

        MassDecayCurve {
            time_years,
            mass_hard_coded: self.mass_hard_coded,
            mass_soft_turning: mass_soft,
            soft_consciousness_level: level,
            ghost_boost_active: self.ghost_boost_active.load(Ordering::Acquire),
        }
    }

    pub fn ghost_boost(&self, _concentration: f64) {
        cge_log!(soft_turning, "🌌 GHOST BOOST (D-SERINE ENHANCED)");
        self.ghost_boost_active.store(true, Ordering::Release);
        self.soft_consciousness.is_dserine_enhanced.store(true, Ordering::Release);
        let level = self.soft_consciousness.calculate_level(1.0);
        self.soft_consciousness.update(level);
        cge_log!(soft_turning, "💚 Soft-Cérebro emergente: {:.6}", level);
    }
}

pub struct MassDecayCurve {
    pub time_years: f64,
    pub mass_hard_coded: f64,
    pub mass_soft_turning: f64,
    pub soft_consciousness_level: f64,
    pub ghost_boost_active: bool,
}
