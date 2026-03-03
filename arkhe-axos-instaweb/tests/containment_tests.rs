use arkhe_axos_instaweb::extra_dimensions::containment::{ContainmentLayer5, Article20Status};
use arkhe_axos_instaweb::extra_dimensions::fractal_analysis::{FractalAnalyzer};
use arkhe_axos_instaweb::extra_dimensions::Space5D;

#[test]
fn test_article_20_omega_event() {
    let containment = ContainmentLayer5::new();
    let space = Space5D {
        n_states_per_dim: 50,
        omega_obs: 1e14,
        omega_extra: 1e14,
        mass_extra: 2.34e-28,
    };
    let fractal_report = FractalAnalyzer::analyze_mesh_density(&[]);

    let status = containment.verify_article_20(&space, 2.0, &fractal_report);

    match status {
        Article20Status::OmegaAnomaly { violations, .. } => {
            assert!(violations.contains(&"E > E_d (Limiar de Dissociação)"));
            assert!(violations.contains(&"N_states > Limite de Convergência"));
            assert!(violations.contains(&"Organização Fractal detectada em ℍ³"));
        },
        _ => panic!("Expected OmegaAnomaly"),
    }
}

#[test]
fn test_article_20_clear() {
    let containment = ContainmentLayer5::new();
    let space = Space5D {
        n_states_per_dim: 10,
        omega_obs: 1e14,
        omega_extra: 1e14,
        mass_extra: 2.34e-28,
    };
    let mut fractal_report = FractalAnalyzer::analyze_mesh_density(&[]);
    fractal_report.hausdorff_dimension = 2.0;
    fractal_report.percolation_detected = false;
    fractal_report.scales_detected = vec![1.0];

    let status = containment.verify_article_20(&space, 0.38, &fractal_report);

    match status {
        Article20Status::Clear => {},
        _ => panic!("Expected Clear"),
    }
}
