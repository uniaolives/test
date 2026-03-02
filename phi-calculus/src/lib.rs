pub const PHI_TARGET: f64 = 1.038;

pub fn calculate_phi_variation(time: f64, fps: f64) -> f64 {
    (time * PHI_TARGET * fps * 57.038).sin() * 0.5 + 0.5
}

pub fn get_frame_phi(time: f64, fps: f64) -> f64 {
    let variation = calculate_phi_variation(time, fps);
    PHI_TARGET + (variation - 0.5) * 0.0005
}
