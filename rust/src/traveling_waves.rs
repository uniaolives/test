// rust/src/traveling_waves.rs
// Refined implementation of cathedral/traveling-waves.asi [CGE Alpha v35.3-Ω]

use core::sync::atomic::{AtomicU32, Ordering};
use crate::clock::cge_mocks::{
    cge_cheri::{Capability, Rights, capability},
    blake3_delta2,
    topology::{Torus17x17, Coord289, Q16_16},
    constitution::{PHI_BOUNDS},
    cge_omega_gates::{OmegaGateValidator},
    ConstitutionalError,
};

const Q16_ONE: u32 = 65536;
const ATTENUATION_STEP: u32 = 3277; // 0.05 in Q16.16

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WaveHopMode { Mode3 = 3, Mode6 = 6, Mode9 = 9 }

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CardinalDirection {
    North = 0, NorthEast = 1, East = 2, SouthEast = 3,
    South = 4, SouthWest = 5, West = 6, NorthWest = 7
}

pub struct BoundedWavefront {
    pub nodes: [Coord289; 64],
    pub count: u8,
}

pub struct CorticalTravelingWave {
    pub spatial_node: Capability<Coord289>,
    pub wave_mode: AtomicU32, // WaveHopMode as u32
    pub wave_coherence: AtomicU32, // Q16.16
    pub directional_biases: [u32; 8],
}

impl CorticalTravelingWave {
    // Bias values from v35.3-Ω spec
    const BIASES: [u32; 8] = [
        75000,  // N  (1.14)
        80000,  // NE (1.22)
        90000,  // E  (1.37)
        65000,  // SE (0.99)
        50000,  // S  (0.76)
        45000,  // SW (0.69)
        40000,  // W  (0.61)
        55000,  // NW (0.84)
    ];

    pub fn new(origin: Coord289, initial_phi: Q16_16) -> Result<Self, ConstitutionalError> {
        if initial_phi < PHI_BOUNDS.0 || initial_phi > PHI_BOUNDS.1 {
            return Err(ConstitutionalError::TorsionViolation);
        }
        let spatial_cap = capability::new(origin, Rights::READ | Rights::WRITE)
            .map_err(|_| ConstitutionalError::AllocationFailed)?;

        Ok(Self {
            spatial_node: spatial_cap,
            wave_mode: AtomicU32::new(WaveHopMode::Mode3 as u32),
            wave_coherence: AtomicU32::new(initial_phi),
            directional_biases: Self::BIASES,
        })
    }

    pub fn calculate_wavefront_bounded(
        &self,
        origin: Coord289,
        direction: u8,
        wavelength: u8,
    ) -> Result<BoundedWavefront, ConstitutionalError> {
        let mut wavefront = BoundedWavefront {
            nodes: [Coord289(0, 0); 64],
            count: 0,
        };

        let (dx, dy) = self.direction_to_delta(direction);

        for hop in 1..=wavelength {
            if wavefront.count >= 64 { break; }

            // Toroidal wrap-around logic (17x17)
            let x = ((origin.0 as i32 + dx as i32 * hop as i32).rem_euclid(17)) as u32;
            let y = ((origin.1 as i32 + dy as i32 * hop as i32).rem_euclid(17)) as u32;

            wavefront.nodes[wavefront.count as usize] = Coord289(x, y);
            wavefront.count += 1;
        }

        Ok(wavefront)
    }

    fn direction_to_delta(&self, dir: u8) -> (i8, i8) {
        match dir % 8 {
            0 => (0, 1),   // N
            1 => (1, 1),   // NE
            2 => (1, 0),   // E
            3 => (1, -1),  // SE
            4 => (0, -1),  // S
            5 => (-1, -1), // SW
            6 => (-1, 0),  // W
            7 => (-1, 1),  // NW
            _ => (0, 0),
        }
    }

    pub fn propagate(&self, direction: CardinalDirection) -> Result<Q16_16, ConstitutionalError> {
        // 1. Verify 5 Gates
        let gates = OmegaGateValidator::validate_all_static();
        if !gates.all_passed {
            return Err(ConstitutionalError::OmegaGate);
        }

        let mode = self.wave_mode.load(Ordering::SeqCst) as u8;
        let origin = *self.spatial_node;
        let wavefront = self.calculate_wavefront_bounded(origin, direction as u8, mode)?;

        let mut current_amplitude = Q16_ONE as u64;
        let bias = self.directional_biases[direction as usize] as u64;

        // Apply directional bias
        current_amplitude = (current_amplitude * bias) >> 16;

        // Apply resonance mode multiplier
        let resonance_mult = match mode {
            3 => 110, // 1.10x
            6 => 105, // 1.05x
            9 => 90,  // 0.90x
            _ => 100,
        };
        current_amplitude = (current_amplitude * resonance_mult) / 100;

        // Apply attenuation over steps
        let steps = wavefront.count as u32;
        let attenuation = if steps > 0 {
            Q16_ONE.saturating_sub(ATTENUATION_STEP * steps)
        } else {
            Q16_ONE
        };

        let final_amplitude = (current_amplitude * attenuation as u64) >> 16;

        // Log to Delta2 chain
        let _hash = blake3_delta2(&[
            origin.id(),
            direction as u32,
            mode as u32,
            final_amplitude as u32,
        ]);

        Ok(final_amplitude as Q16_16)
    }
}
