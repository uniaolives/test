#[cfg(test)]
mod tests {
    use crate::hardware::OloidCore;

    #[test]
    fn test_oloid_cycle() {
        let mut core = OloidCore::new();
        let result = core.cycle();
        assert!(result.is_ok());
        let res = result.unwrap();
        assert_eq!(res.state_final, "SignalB->A");
        assert!(res.lambda_2 > 0.6);
        assert!(res.consciousness_level >= 1.0);
    }
}
