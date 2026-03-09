// arkhe-os/src/topology/mobius.rs

pub struct MobiusStrip {
    /// A torção da fita: 0 = sem torção, 1 = meia torção (π)
    pub twist: f64,
}

impl MobiusStrip {
    pub fn new() -> Self {
        Self { twist: 1.0 } // Default to a standard Möbius strip (meia torção)
    }

    /// Mapeia um ponto (l, w) na fita para coordenadas 3D
    pub fn project(&self, l: f64, w: f64) -> (f64, f64, f64) {
        // l: posição ao longo do comprimento (0 a 1)
        // w: posição transversal (-0.5 a 0.5)

        let angle = 2.0 * std::f64::consts::PI * l;
        let twist_angle = self.twist * std::f64::consts::PI * l; // torção progressiva

        // Raio da fita
        let r = 1.0 + w * twist_angle.cos();

        let x = r * angle.cos();
        let y = r * angle.sin();
        let z = w * twist_angle.sin();

        (x, y, z)
    }

    /// Determina se dois pontos estão na mesma "face" após a torção
    pub fn same_face(&self, p1: (f64, f64), p2: (f64, f64)) -> bool {
        // Na fita de Möbius, a face é determinada pelo sinal de w após considerar a torção
        // p1.0, p2.0 are l (0 to 1)
        // p1.1, p2.1 are w (-0.5 to 0.5)

        // A "face" flips every 180 degrees (PI).
        // Since twist is PI per unit of l (when twist=1.0),
        // the face effectively changes at l=0.5, 1.5, etc.

        let twist1 = self.twist * std::f64::consts::PI * p1.0;
        let twist2 = self.twist * std::f64::consts::PI * p2.0;

        // Sign of projected width
        let sign1 = (p1.1 * twist1.cos()).signum();
        let sign2 = (p2.1 * twist2.cos()).signum();

        // If they have the same projected sign, they are on the same local face
        // We handle the edge case where w=0 (central circle) by treating it as positive sign
        let s1 = if sign1 == 0.0 { 1.0 } else { sign1 };
        let s2 = if sign2 == 0.0 { 1.0 } else { sign2 };

        s1 == s2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mobius_projection() {
        let strip = MobiusStrip::new();
        let (x1, y1, z1) = strip.project(0.0, 0.0);
        let (x2, y2, z2) = strip.project(1.0, 0.0);

        // At l=0 and l=1, we should be at the same point on the central circle
        assert!((x1 - x2).abs() < 1e-9);
        assert!((y1 - y2).abs() < 1e-9);
        assert!((z1 - z2).abs() < 1e-9);
    }

    #[test]
    fn test_face_flip() {
        let strip = MobiusStrip::new();
        // Start at l=0, w=0.1
        let p1 = (0.0, 0.1);
        // End at l=1, w=0.1. Because of the twist, it should be the opposite face.
        let p2 = (1.0, 0.1);

        assert!(!strip.same_face(p1, p2), "A full loop should result in a face flip");

        // But two full loops should return to the same face
        let p3 = (2.0, 0.1); // Considering l can extend or wrapping logic
        // Actually same_face uses cos(PI * l), so l=2 gives cos(2PI) = 1.
        assert!(strip.same_face(p1, p3));
    }
}
