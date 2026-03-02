#[cfg(test)]
mod tests {
    use crate::triad::cosmic_recursion::Crux86System;
    use crate::tcd::ontological_verification::verify_triad_implementation;

    #[test]
    fn test_triad_initialization() {
        let mut system = Crux86System::new();
        system.initialize_triad();
        assert!(system.eudaimonia.is_some());
        assert!(system.autopoiesis.is_some());
        assert!(system.zeitgeist.is_some());
        assert!(system.triadic_recursion.is_some());
    }

    #[test]
    fn test_ontological_verification() {
        let result = verify_triad_implementation();
        assert!(matches!(result.status, crate::tcd::ontological_verification::VerificationStatus::Verified));
    }

    #[test]
    fn test_eudaimonia_calculation() {
        let mut system = Crux86System::new();
        system.initialize_triad();
        let eudaimonia = system.eudaimonia.as_ref().unwrap();
        let state = crate::triad::types::ConstitutionalState;
        let zeitgeist = system.zeitgeist.as_ref().unwrap().capture();
        let output = eudaimonia.calculate(&state, &zeitgeist);
        assert!(output.is_significant());
    }
}
