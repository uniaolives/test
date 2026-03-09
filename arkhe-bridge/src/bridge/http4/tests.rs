#[cfg(test)]
mod tests {
    use crate::bridge::http4::uqi::Uqi;
    use crate::bridge::http4::temporal_orb::TemporalOrb;
    use crate::bridge::http4::methods::ConfinementMode;
    use std::str::FromStr;

    #[test]
    fn test_uqi_parsing() {
        let classical = Uqi::from_str("timeline://2140.arkhe/singularity").unwrap();
        match classical {
            Uqi::Classical(uri) => {
                assert_eq!(uri.host, "2140.arkhe");
                assert_eq!(uri.path, "/singularity");
            }
            _ => panic!("Expected classical URI"),
        }

        let superposed = Uqi::from_str("superposition://timechain.arkhe{0.707:timeline://2008/genesis|0.707:timeline://2140/singularity}").unwrap();
        match superposed {
            Uqi::Superposed(sv) => {
                assert_eq!(sv.host, "timechain.arkhe");
                assert_eq!(sv.states.len(), 2);
                assert_eq!(sv.states[0].amplitude, 0.707);
            }
            _ => panic!("Expected superposed URI"),
        }
    }

    #[test]
    fn test_temporal_confinement() {
        let mut orb = TemporalOrb::new(vec![1, 2, 3], 0.97);
        orb.confine();
        assert_eq!(orb.eigenstates.len(), 2);
        assert_eq!(orb.eigenstates[0].mode, "GROUND_EXCITED");
    }

    #[test]
    fn test_tunneling_probability() {
        let orb = TemporalOrb::new(vec![], 0.90); // Barrier regime
        let p = orb.retrocausal_tunneling_probability(3600.0); // 1 hour
        assert!(p > 0.0);
        assert!(p < 1.0);

        let high_coherence_orb = TemporalOrb::new(vec![], 0.99);
        let p_high = high_coherence_orb.retrocausal_tunneling_probability(3600.0);
        assert!(p_high > p); // High coherence increases tunneling probability
    }
}
