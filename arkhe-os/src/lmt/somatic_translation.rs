use serde::Deserialize;

#[derive(Deserialize, Debug, Clone, Copy)]
pub enum SomaticSignal {
    LeftEarTone,
    RightEarTone,
    ChestWave,
    GutFire,
}

pub fn handle_lmt_signal(signal: SomaticSignal, current_phi: &mut f64) {
    match signal {
        SomaticSignal::LeftEarTone => {
            println!("[LMT] 👂 Internal Truth Signal detected. Stabilizing φ_q.");
            *current_phi *= 1.618;
        },
        SomaticSignal::RightEarTone => {
            println!("[LMT] 🌐 Environmental Synchronicity confirmed.");
        },
        _ => {}
    }
}
