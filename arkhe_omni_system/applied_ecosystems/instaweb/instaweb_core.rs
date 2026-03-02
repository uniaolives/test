//! instaweb_core.rs
//! Núcleo de transmissão símbolo-síncrona para latência zero
//!
//! Baseado em: RF-Zero-Wire e ZERO-WIRE

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::collections::VecDeque;
use tokio::sync::mpsc;
use tokio::time::{Instant, Duration};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use num_complex::Complex;
use rustfft::{Fft, FftPlanner, FftDirection};

// Constantes fundamentais
const MAX_NETWORK_RADIUS: f64 = 20000.0; // 20,000 km (Half Earth)
const FFT_SIZE: usize = 256;
const DATA_SUBCARRIERS: usize = 128;
const DC_BIAS: f64 = 0.5;
const BATCH_SIZE: usize = 512;
const MAX_THREADS: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Symbol {
    Zero = 0,
    One = 1,
}

pub type NodeId = u64;

#[derive(Clone, Copy, Debug)]
pub struct HyperbolicCoord {
    pub r: Decimal,
    pub theta: Decimal,
    pub z: Decimal,
}

/// DCO-OFDM Symbol Mapper para OWC
pub struct DcoOfdmModulator {
    fft_planner: Arc<dyn Fft<f64>>,
    dc_bias: f64,
    clip_threshold: f64,
}

impl DcoOfdmModulator {
    pub fn new() -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft(FFT_SIZE, FftDirection::Inverse);
        Self {
            fft_planner: fft,
            dc_bias: DC_BIAS,
            clip_threshold: 1.0,
        }
    }

    pub fn modulate(&self, bits: &[u8]) -> Vec<f64> {
        let qam_symbols: Vec<Complex<f64>> = bits
            .chunks(4)
            .map(|chunk| {
                let val = chunk.iter().fold(0, |acc, &b| (acc << 1) | b as i32);
                Complex::new(val as f64 / 15.0, 0.0)
            })
            .collect();

        let mut freq_domain = vec![Complex::new(0.0, 0.0); FFT_SIZE];
        for (i, &sym) in qam_symbols.iter().take(DATA_SUBCARRIERS).enumerate() {
            freq_domain[i + 1] = sym;
            freq_domain[FFT_SIZE - 1 - i] = sym.conj();
        }

        let mut time_domain = freq_domain.clone();
        self.fft_planner.process(&mut time_domain);

        time_domain.iter()
            .map(|c| {
                let real = c.re + self.dc_bias;
                real.min(self.clip_threshold).max(0.0)
            })
            .collect()
    }
}

/// Embedding robusto com evitação de singularidade
/// Converte coordenadas geográficas (Lat, Lon, Alt) para o Disco de Poincaré ℍ³
pub fn geo_to_poincare(lat: f64, lon: f64, alt_m: f64,
                       network_center: (f64, f64)) -> HyperbolicCoord {
    // 1. Distância geodésica ao centro da rede (Haversine approximation)
    let d_geo = haversine_dist((lat, lon), network_center);

    // 2. Normalização: mapear [0, D_max] para [0, 0.95] (evitar borda do disco)
    let r = (d_geo / MAX_NETWORK_RADIUS).tanh() * 0.95;

    // 3. Ângulo preservando bearing relativo ao centro
    let theta = bearing_to_center(network_center, (lat, lon)).to_radians();

    // 4. Altitude como coordenada z (compactificada)
    let z = (alt_m / 10000.0).tanh(); // 10km -> ~1.0

    HyperbolicCoord {
        r: Decimal::from_f64(r).unwrap(),
        theta: Decimal::from_f64(theta).unwrap(),
        z: Decimal::from_f64(z).unwrap()
    }
}

fn haversine_dist(p1: (f64, f64), p2: (f64, f64)) -> f64 {
    let earth_radius = 6371.0;
    let d_lat = (p2.0 - p1.0).to_radians();
    let d_lon = (p2.1 - p1.1).to_radians();
    let a = (d_lat/2.0).sin() * (d_lat/2.0).sin() +
            p1.0.to_radians().cos() * p2.0.to_radians().cos() *
            (d_lon/2.0).sin() * (d_lon/2.0).sin();
    let c = 2.0 * a.sqrt().atan2((1.0-a).sqrt());
    earth_radius * c
}

fn bearing_to_center(center: (f64, f64), pos: (f64, f64)) -> f64 {
    let d_lon = (pos.1 - center.1).to_radians();
    let y = d_lon.sin() * pos.0.to_radians().cos();
    let x = center.0.to_radians().cos() * pos.0.to_radians().sin() -
            center.0.to_radians().sin() * pos.0.to_radians().cos() * d_lon.cos();
    y.atan2(x).to_degrees()
}

pub struct InstaNode {
    pub id: NodeId,
    pub neighbors: Vec<NodeId>,
    pub hyper_coords: HyperbolicCoord,
}

impl InstaNode {
    pub fn hyperbolic_distance(&self, other: &HyperbolicCoord) -> f64 {
        let r1 = self.hyper_coords.r.to_f64().unwrap();
        let th1 = self.hyper_coords.theta.to_f64().unwrap();
        let z1 = self.hyper_coords.z.to_f64().unwrap();

        let r2 = other.r.to_f64().unwrap();
        let th2 = other.theta.to_f64().unwrap();
        let z2 = other.z.to_f64().unwrap();

        let dr = r1 - r2;
        let dth = (th1 - th2).abs();
        let dz = z1 - z2;

        let numerator = dr * dr + (r1 * r2 * (1.0 - dth.cos())) + dz * dz;
        let denominator = 2.0 * z1 * z2;

        let arg = 1.0 + numerator / denominator;
        arg.max(1.0).acosh()
    }
}
