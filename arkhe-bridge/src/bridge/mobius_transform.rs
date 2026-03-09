// src/bridge/mobius_transform.rs

pub struct MobiusTransform {
    pub twist_factor: f64,  // 0.5 = meia torção (Möbius padrão)
}

impl MobiusTransform {
    pub fn new() -> Self {
        Self { twist_factor: 0.5 }
    }

    /// Transforma um tempo linear (t) em coordenadas na fita de Möbius
    pub fn linear_to_mobius(&self, t: f64, w: f64) -> (f64, f64) {
        // t: tempo linear (0 = passado remoto, 1 = futuro remoto)
        // w: "largura" da informação (coerência, etc.)

        // Aplica a torção: a face muda quando t cruza 0.5
        let twisted_t = t;
        let twisted_w = w * (self.twist_factor * 2.0 * std::f64::consts::PI * t).cos();

        (twisted_t, twisted_w)
    }

    /// Verifica se dois eventos estão na mesma "folha" temporal
    pub fn same_temporal_slice(&self, event1: (f64, f64), event2: (f64, f64)) -> bool {
        let (t1, w1) = self.linear_to_mobius(event1.0, event1.1);
        let (t2, w2) = self.linear_to_mobius(event2.0, event2.1);

        // Na fita de Möbius, a "foliação" (slice) é definida por t constante,
        // mas w pode ter sinais diferentes dependendo da torção.
        // Dois eventos estão no mesmo slice temporal se t1 == t2 E o sinal de w é o mesmo.
        (t1 - t2).abs() < 1e-6 && (w1 * w2) >= 0.0 // Changed to >= to handle zero
    }
}
