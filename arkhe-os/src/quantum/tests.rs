use crate::quantum::berry::{TopologicalQubit, SpinState};

#[test]
fn test_helical_switch_logic() {
    let mut tq = TopologicalQubit::new();
    assert_eq!(tq.spin_state, SpinState::Triplet);

    // Low stimulus -> stays Triplet
    tq.trigger_helical_switch(1.0);
    assert_eq!(tq.spin_state, SpinState::Triplet);

    // High stimulus -> switches to Singlet (Heloidal)
    tq.trigger_helical_switch(5.0);
    assert!(tq.spin_state == SpinState::SingletP || tq.spin_state == SpinState::SingletM);
}

#[test]
fn test_berry_phase_periodicity() {
    let mut tq = TopologicalQubit::new();
    for _ in 0..8 {
        tq.circumnavigate();
    }
    // 8 turns = 2pi = 0 (mod 2pi) since each turn is pi/4
    assert!(tq.is_coherent());
    assert!(tq.berry_phase.abs() < 0.001 || (tq.berry_phase - 6.283).abs() < 0.001);
}
