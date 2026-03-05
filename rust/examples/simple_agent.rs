// rust/examples/simple_agent.rs
use sasc_core::memory::dmr::*;
use std::time::Duration;

fn main() {
    println!("--- Digital Memory Ring (DMR) Demonstration ---");

    let vk_ref = KatharosVector::new(0.5, 0.5, 0.5, 0.5);
    let formation_interval = Duration::from_secs(3600); // 1 hour
    let mut dmr = DigitalMemoryRing::new("demonstration-agent-01".to_string(), vk_ref, formation_interval);

    println!("Simulating 7 days of agent activity...");

    // Day 1-3: Stability
    for day in 1..=3 {
        println!("Day {}: Stable", day);
        for _ in 0..24 {
            let state = SystemState {
                vk: KatharosVector::new(0.5, 0.5, 0.5, 0.5),
                entropy: 0.1,
                events: Vec::new(),
            };
            dmr.grow_layer(state).unwrap();
        }
    }

    // Day 4: Crisis
    println!("Day 4: Crisis Induced");
    for _ in 0..24 {
        let state = SystemState {
            vk: KatharosVector::new(0.9, 0.9, 0.9, 0.9),
            entropy: 0.8,
            events: vec![CellularEvent { event_type: "STRESS".to_string(), metadata: "External perturbation".to_string() }],
        };
        dmr.grow_layer(state).unwrap();
    }

    // Day 5-7: Recovery
    for day in 5..=7 {
        println!("Day {}: Recovering", day);
        for _ in 0..24 {
            let state = SystemState {
                vk: KatharosVector::new(0.55, 0.55, 0.55, 0.55),
                entropy: 0.3,
                events: Vec::new(),
            };
            dmr.grow_layer(state).unwrap();
        }
    }

    println!("--- Analysis ---");
    let t_kr = dmr.measure_t_kr();
    println!("Total t_KR accumulated: {} hours", t_kr.as_secs() / 3600);

    let bifurcations = &dmr.bifurcations;
    println!("Number of bifurcations detected: {}", bifurcations.len());
    for bif in bifurcations {
        println!("  - {:?} at ΔK={:.4}", bif.bifurcation_type, bif.delta_k);
    }

    let periods = dmr.find_katharos_periods();
    println!("Number of stable periods: {}", periods.len());

    let trajectory = dmr.reconstruct_trajectory();
    println!("Trajectory reconstructed with {} layers.", trajectory.timestamps.len());

    println!("--- Demo Complete ---");
}
