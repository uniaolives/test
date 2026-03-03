use rayon::prelude::*;
use rand::Rng;
use std::time::Instant;

const A: f64 = 0.95;
const B: f64 = 0.7;
const C: f64 = 0.6;
const D: f64 = 3.5;
const E: f64 = 0.25;
const F: f64 = 0.1;

#[derive(Clone, Copy, Debug)]
struct State {
    x: f64,
    y: f64,
    z: f64,
}

#[inline(always)]
fn derivatives(s: State) -> (f64, f64, f64) {
    let dx = (s.z - B) * s.x - D * s.y;
    let dy = D * s.x + (s.z - B) * s.y;
    let dz = C + A * s.z - (s.z.powi(3)) / 3.0 - (s.x.powi(2) + s.y.powi(2)) * (1.0 + E * s.z) + F * s.z * (s.x.powi(3));
    (dx, dy, dz)
}

#[inline(always)]
fn rk4_step(s: State, dt: f64) -> State {
    let (k1x, k1y, k1z) = derivatives(s);

    let (k2x, k2y, k2z) = derivatives(State {
        x: s.x + 0.5 * dt * k1x,
        y: s.y + 0.5 * dt * k1y,
        z: s.z + 0.5 * dt * k1z,
    });

    let (k3x, k3y, k3z) = derivatives(State {
        x: s.x + 0.5 * dt * k2x,
        y: s.y + 0.5 * dt * k2y,
        z: s.z + 0.5 * dt * k2z,
    });

    let (k4x, k4y, k4z) = derivatives(State {
        x: s.x + dt * k3x,
        y: s.y + dt * k3y,
        z: s.z + dt * k3z,
    });

    State {
        x: s.x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x),
        y: s.y + (dt / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y),
        z: s.z + (dt / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z),
    }
}

fn main() {
    let n_points = 1_000_000;
    let steps = 100;
    let dt = 0.01;

    let mut rng = rand::thread_rng();
    let initial_states: Vec<State> = (0..n_points)
        .map(|_| State {
            x: rng.gen_range(-1.0..1.0),
            y: rng.gen_range(-1.0..1.0),
            z: rng.gen_range(-1.0..1.0),
        })
        .collect();

    println!("Starting Aizawa Rust Rayon Simulation with {} points for {} steps...", n_points, steps);

    let start = Instant::now();
    let _final_states: Vec<State> = initial_states
        .into_par_iter()
        .map(|mut s| {
            for _ in 0..steps {
                s = rk4_step(s, dt);
            }
            s
        })
        .collect();
    let duration = start.elapsed();

    println!("Simulation completed in {:.4} seconds.", duration.as_secs_f64());
    println!("Performance: {:.2} million iterations per second.", (n_points as f64 * steps as f64) / duration.as_secs_f64() / 1e6);
}
