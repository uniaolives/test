use sasc_core::cognition::complex_engine::ComplexInferenceEngine;
use sasc_core::stability::spectral_monitor::SpectralStabilityMonitor;
use num_complex::Complex64;

#[test]
fn test_complex_xor_logic() {
    // Codificação bipolar: bit 0 -> -1.0, bit 1 -> 1.0
    // XOR(x, y) é verdadeiro se o produto x*y < 0

    let xor = |x: f64, y: f64| {
        let z = Complex64::new(x, y);
        (z.re * z.im) < 0.0
    };

    assert_eq!(xor(1.0, 1.0), false);   // 1 XOR 1 = 0
    assert_eq!(xor(1.0, -1.0), true);   // 1 XOR 0 = 1
    assert_eq!(xor(-1.0, 1.0), true);   // 0 XOR 1 = 1
    assert_eq!(xor(-1.0, -1.0), false); // 0 XOR 0 = 0
}

#[test]
fn test_spectral_stability() {
    let mut monitor = SpectralStabilityMonitor::new(10);

    // Caso estável (dentro do círculo unitário)
    monitor.record_eigenvalues(vec![Complex64::new(0.5, 0.0)]);
    assert!(monitor.check_druj_alert().is_none());

    // Caso Druj (fora do círculo unitário)
    monitor.record_eigenvalues(vec![Complex64::new(1.1, 0.0)]);
    assert!(monitor.check_druj_alert().is_some());
}
