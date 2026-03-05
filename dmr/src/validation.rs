//! Three validation experiments: DMR‑1, DMR‑2, DMR‑3.

use crate::{DigitalMemoryRing, KatharosVector, analysis::VKTrajectory};
use rand::Rng;
use std::time::Duration;

/// Run all three experiments and print summary.
pub fn run_validation_suite() {
    println!("=== Digital Memory Ring Validation Suite ===");
    dmr1_linearity();
    dmr2_bifurcation_detection();
    dmr3_gemini_pattern();
}

/// DMR‑1: t_KR accumulates linearly in stable state.
fn dmr1_linearity() {
    println!("\n--- DMR‑1: t_KR Linearity ---");
    let vk_ref = KatharosVector::new(0.5, 0.5, 0.5, 0.5);
    let mut ring = DigitalMemoryRing::new(
        "test-1".to_string(),
        vk_ref.clone(),
        Duration::from_secs(3600), // 1 hour
    );

    // Simulate 24 hours in stable state (ΔK < 0.3)
    let stable_vk = KatharosVector::new(0.55, 0.52, 0.48, 0.51); // ΔK ~ 0.07
    for _ in 0..24 {
        ring.grow_layer(stable_vk.clone(), 0.9, vec![]);
    }

    let expected_tkr = Duration::from_secs(24 * 3600);
    let actual_tkr = ring.t_kr;
    let diff = if actual_tkr > expected_tkr {
        actual_tkr - expected_tkr
    } else {
        expected_tkr - actual_tkr
    };

    println!("Expected t_KR: {:?}", expected_tkr);
    println!("Actual t_KR:   {:?}", actual_tkr);
    println!("Difference:    {:?}", diff);
    assert!(diff < Duration::from_secs(1), "t_KR should be exactly 24 hours ± epsilon");
    println!("✅ DMR‑1 passed.");
}

/// DMR‑2: Bifurcation detection.
fn dmr2_bifurcation_detection() {
    println!("\n--- DMR‑2: Bifurcation Detection ---");
    let vk_ref = KatharosVector::new(0.5, 0.5, 0.5, 0.5);
    let mut ring = DigitalMemoryRing::new(
        "test-2".to_string(),
        vk_ref.clone(),
        Duration::from_secs(3600),
    );

    // 10 hours stable
    let stable = KatharosVector::new(0.52, 0.51, 0.49, 0.5);
    for _ in 0..10 {
        ring.grow_layer(stable.clone(), 0.9, vec![]);
    }

    // 2 hours crisis (ΔK > 0.7)
    let crisis = KatharosVector::new(1.2, 0.3, 0.2, 0.1); // ΔK >> 0.7
    for _ in 0..2 {
        ring.grow_layer(crisis.clone(), 0.2, vec!["crisis".to_string()]);
    }

    // 12 hours back to stable
    for _ in 0..12 {
        ring.grow_layer(stable.clone(), 0.9, vec![]);
    }

    println!("Bifurcations detected: {:?}", ring.bifurcations);
    // Should have at least one entry and one exit
    assert!(!ring.bifurcations.is_empty(), "No bifurcation detected");
    println!("✅ DMR‑2 passed.");
}

/// DMR‑3: Replicate GEMINI NFκB pattern (simulated).
fn dmr3_gemini_pattern() {
    println!("\n--- DMR‑3: GEMINI Pattern Replication ---");
    let vk_ref = KatharosVector::new(0.5, 0.5, 0.5, 0.5);
    let mut ring = DigitalMemoryRing::new(
        "test-3".to_string(),
        vk_ref.clone(),
        Duration::from_secs(900), // 15 minutes (fast dynamics)
    );

    // Simulate TNF-α pulse: ΔK rises and falls.
    let mut rng = rand::thread_rng();
    let mut gemini_sim = Vec::new();

    for t in 0..48 { // 12 hours with 15‑min steps
        // Create a smooth pulse: peak around t=16 (4 hours)
        let pulse = if t > 8 && t < 24 {
            let peak = 16.0;
            let dist = (t as f64 - peak).abs();
            0.7 * (-dist * dist / 20.0).exp() // Gaussian
        } else {
            0.0
        };
        // ΔK = base + pulse + noise
        let base_dk = 0.15;
        let noise: f64 = rng.gen_range(-0.02..0.02);
        let _delta_k = base_dk + pulse + noise;

        // Build a VK that yields this ΔK (approximate)
        let vk = KatharosVector::new(
            0.5 + 0.3 * pulse,
            0.5 - 0.1 * pulse,
            0.5 + 0.05 * pulse,
            0.5,
        );

        ring.grow_layer(vk, 0.8 - 0.2 * pulse, vec![]);
        gemini_sim.push(pulse); // store simulated GEMINI intensity
    }

    let traj = VKTrajectory::from_ring(&ring);
    let correlation = traj.compare_with_gemini(&gemini_sim);

    println!("Pearson correlation with GEMINI pattern: {:.3}", correlation);
    assert!(correlation > 0.85, "Correlation should exceed 0.85");
    println!("✅ DMR‑3 passed.");
}
