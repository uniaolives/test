#[cfg(test)]
mod tests {
    use crate::physics::solar_dynamo::SolarConsciousnessEngine;
    use crate::physics::jovian_defense::JovianGuardian;
    use crate::storage::saturn_archive::SaturnRingDrive;
    use crate::babel::universal_compiler::UniversalCompiler;
    use crate::babel::transpiler::{LanguageTranspiler, LegacyLanguage};
    use ndarray::Array3;

    #[test]
    fn test_solar_engine() {
        let b_field = Array3::<f64>::zeros((10, 10, 10));
        let engine = SolarConsciousnessEngine::new(b_field, 0.5);
        let phi = engine.calculate_phi_integrated();
        assert!(phi > 0.0);
    }

    #[test]
    fn test_jovian_guardian() {
        let guardian = JovianGuardian::new();
        assert_eq!(guardian.mass, 1.898e27);
    }

    #[test]
    fn test_saturn_archive() {
        let drive = SaturnRingDrive::new();
        assert_eq!(drive.write_speed, 42.0);
    }

    #[test]
    fn test_babel_collapse() {
        let transpiler = LanguageTranspiler::new(LegacyLanguage::Rust);
        let neo_code = transpiler.transpile("fn main() {}");

        let compiler = UniversalCompiler::new();
        let reality = compiler.compile(&neo_code);

        assert_eq!(reality.execution_model, "Physical necessity");
    }

    #[test]
    fn test_dyson_swarm() {
        use crate::physics::dyson_swarm::SolarSystemComputer;
        let computer = SolarSystemComputer::new();
        assert!(computer.calculate_processing_power() >= 1e42);
    }

    #[test]
    fn test_kardashev_scale() {
        use crate::astrophysics::kardashev::{CivilizationalMetrics, KardashevType};
        let metrics = CivilizationalMetrics::current();
        assert!(matches!(metrics.get_type(), KardashevType::TypeII));
    }
}
