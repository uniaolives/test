use arkhe_axos_instaweb::extra_dimensions::spectrum_prediction::KaluzaKleinSpectrum;
use arkhe_axos_instaweb::extra_dimensions::portal_threshold::PortalAnalysis;

#[test]
fn test_spectrum_prediction() {
    let spectrum = KaluzaKleinSpectrum::new_from_detection();
    let transitions = spectrum.predict_transitions(5);
    assert_eq!(transitions.len(), 5);
    assert_eq!(transitions[0].frequency_ghz, 147034.0);
    assert_eq!(transitions[1].frequency_ghz, 294068.0);
}

#[test]
fn test_portal_threshold() {
    let portal = PortalAnalysis::from_detected_resonance();
    let threshold = portal.dissociation_threshold();
    // Expected around 1.897 eV
    assert!(threshold > 1.8 && threshold < 2.0);

    let safety = portal.safety_assessment(0.380); // n=1
    match safety {
        arkhe_axos_instaweb::extra_dimensions::portal_threshold::SafetyRating::Safe { .. } => {},
        _ => panic!("Should be safe"),
    }
}
