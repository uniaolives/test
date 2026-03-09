// arkhe-os/src/topology/causal_graph.rs

use crate::topology::mobius::MobiusStrip;

pub struct CausalEvent {
    pub layer: f64,
    pub position: f64,
    pub payload: String,
}

pub struct TwistedCausalGraph {
    pub mobius: MobiusStrip,
    pub num_layers: usize,
    pub seq_len: usize,
}

impl TwistedCausalGraph {
    pub fn new(num_layers: usize, seq_len: usize) -> Self {
        Self {
            mobius: MobiusStrip::new(),
            num_layers,
            seq_len,
        }
    }

    /// Maps a layer and position to (l, w) coordinates on the Möbius strip
    /// layer -> width (transversal)
    /// position -> length (longitudinal)
    pub fn map_to_mobius(&self, layer: usize, position: usize) -> (f64, f64) {
        let l = position as f64 / self.seq_len as f64;
        let w = (layer as f64 / self.num_layers as f64) - 0.5;
        (l, w)
    }

    /// Determines if there is a causal connection between two events.
    /// In a normal transformer, causality is position1 <= position2.
    /// In the twisted graph, we consider the Möbius topology.
    pub fn is_connected(&self, e1: &CausalEvent, e2: &CausalEvent) -> bool {
        let p1 = self.map_to_mobius(e1.layer as usize, e1.position as usize);
        let p2 = self.map_to_mobius(e2.layer as usize, e2.position as usize);

        // On a Möbius strip, the faces are "locally" distinct but globally one.
        // We use the `same_face` helper to determine if we are in a "normal" or "twisted" regime.

        let same_face = self.mobius.same_face(p1, p2);

        if same_face {
            // Locally same face: cause MUST precede effect
            e1.position <= e2.position
        } else {
            // Globally same face, but locally "flipped":
            // This represents a path that has crossed the 180 degree twist.
            // In this regime, the "future" can connect to the "past".
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_twisted_causality() {
        let graph = TwistedCausalGraph::new(12, 128);

        // Use positions that are locally on the same face (both < 0.5 seq_len)
        let e_past = CausalEvent {
            layer: 0.0,
            position: 10.0,
            payload: "past".to_string(),
        };

        let e_near_future = CausalEvent {
            layer: 0.0,
            position: 40.0,
            payload: "near_future".to_string(),
        };

        // Normal causal direction
        assert!(graph.is_connected(&e_past, &e_near_future));

        // Normal retrocausal direction (should be false on same face)
        assert!(!graph.is_connected(&e_near_future, &e_past));

        // Let's force a connection by crossing the twist
        // e_past on one "side" of the strip, e_future on the "other" (after twist)
        // position 0 and position 128 (max) are near the twist.

        let e_loopback = CausalEvent {
            layer: 0.0,
            position: 128.0, // Future end
            payload: "loopback".to_string(),
        };

        let e_origin = CausalEvent {
            layer: 0.0,
            position: 0.0, // Past start
            payload: "origin".to_string(),
        };

        // Twisted connectivity: if they are on different faces (via twist), they are connected.
        // position 0: cos(0) = 1
        // position 128: cos(PI * 128/128) = -1
        assert!(graph.is_connected(&e_loopback, &e_origin), "Retrocausal jump should be possible through the twist");
    }
}
