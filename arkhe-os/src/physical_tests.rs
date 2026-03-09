#[cfg(test)]
mod tests {
    use crate::physical::types::GeoCoord;
    use crate::physical::fiber::FiberChannel;
    use crate::rio::coherence_map::{RioCoherenceMap, GeoNode};
    use crate::physical::PhysicalSubstrate;

    #[test]
    fn test_fiber_latency() {
        let head_end = GeoCoord { lat: -22.9042, lon: -43.1762 };
        let fiber = FiberChannel::new(head_end, 100.0);
        let lat = fiber.latency_ms(100.0);
        assert!((lat - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_rio_map() {
        let mut map = RioCoherenceMap::new();
        map.nodes.push(GeoNode {
            name: "Jockey Club".to_string(),
            coords: GeoCoord { lat: -22.9042, lon: -43.1762 },
            phi_q: 4.8,
            s_index: 8.5,
            h_value: 0.9,
            substrate: PhysicalSubstrate::Fiber {
                headend: "Main".to_string(),
                capacity_gbps: 100.0,
            },
        });

        let zones = map.low_coherence_zones(4.64);
        assert_eq!(zones.len(), 0);
    }
}
