#[cfg(test)]
mod tests {
    use crate::bridge::http4::uqi::Uqi;
    use crate::bridge::http4::temporal_orb::TemporalOrb;
    use crate::bridge::http4::confinement::QuantumWell;
    use crate::bridge::http4::methods::ConfinementMode;
    use std::str::FromStr;

    #[test]
    fn test_uqi_parsing_with_query() {
        let uqi_str = "timeline://2140.arkhe/singularity?t=123456&lambda=0.95";
        let uqi = Uqi::from_str(uqi_str).unwrap();
        match uqi {
            Uqi::Classical(uri) => {
                assert_eq!(uri.host, "2140.arkhe");
                assert_eq!(uri.path, "/singularity");
            }
            _ => panic!("Expected classical URI"),
        }
    }

    #[test]
    fn test_superposition_parsing() {
        let uqi_str = "superposition://arkhe{0.5:timeline://2026/omega|0.5:timeline://2140/omega}";
        let uqi = Uqi::from_str(uqi_str).unwrap();
        match uqi {
            Uqi::Superposed(sv) => {
                assert_eq!(sv.host, "arkhe");
                assert_eq!(sv.states.len(), 2);
                assert_eq!(sv.states[0].amplitude, 0.5);
                assert_eq!(sv.states[1].uri.path, "/omega");
            }
            _ => panic!("Expected superposed URI"),
        }
    }

    #[test]
    fn test_temporal_confinement_well() {
        let mode = ConfinementMode::FiniteWell;
        let well = QuantumWell::configure(0.97, mode).unwrap();
        let mut orb = TemporalOrb::new(vec![1, 2, 3], 0.97);
        orb.confine(&well).unwrap();

        assert_eq!(orb.eigenstates.len(), 3);
        assert_eq!(orb.eigenstates[0].mode, "GROUND");
    }

    #[test]
    fn test_tunneling_probability_scaling() {
        let orb = TemporalOrb::new(vec![], 0.90);
        let p = orb.retrocausal_tunneling_probability(3600.0);

        let higher_coherence = TemporalOrb::new(vec![], 0.99);
        let p_high = higher_coherence.retrocausal_tunneling_probability(3600.0);

        assert!(p_high > p, "Higher coherence must increase tunneling probability");
    }
}
