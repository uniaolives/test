// modules/orbital/quantum/doppler_compensation.rs
// ASI-Sat: Doppler Compensation for Laser Handovers

pub struct DopplerCompensator {
    pub laser_frequency_thz: f64,
}

impl DopplerCompensator {
    pub fn new() -> Self {
        Self { laser_frequency_thz: 193.5 }
    }

    /// Compute Doppler shift for inter-satellite link
    pub fn compute_shift(&self, relative_velocity_ms: f64) -> f64 {
        let speed_of_light = 299792458.0;
        let beta = relative_velocity_ms / speed_of_light;

        // Return shift in GHz
        self.laser_frequency_thz * beta * 1000.0
    }

    pub fn compensate(&self, shift_ghz: f64) -> (f64, f64) {
        // Return (tx_offset, rx_offset) in GHz
        (-shift_ghz, shift_ghz)
    }
}
