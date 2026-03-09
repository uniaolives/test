// arkhe_orb_core.rs
use std::f64::consts::PI;

pub struct Orb {
    stability: f64,
    throat: WormholeThroat,
}

pub struct WormholeThroat {
    entrance: (f64, f64), // (lat, lon)
    exit: (f64, f64),
    bandwidth_hz: f64,
}

impl Orb {
    pub fn new(lambda: f64, rf_freq: f64) -> Option<Self> {
        if lambda > 0.618 && rf_freq > 1e9 {
            Some(Self {
                stability: lambda,
                throat: WormholeThroat {
                    entrance: (0.0, 0.0),
                    exit: (0.0, 0.0),
                    bandwidth_hz: rf_freq,
                },
            })
        } else {
            None
        }
    }

    pub fn transmit(&self, handover: &[u8]) -> Result<(), &'static str> {
        if self.stability > 0.5 {
            Ok(())
        } else {
            Err("Wormhole collapsed")
        }
    }
}
