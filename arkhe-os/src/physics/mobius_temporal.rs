// src/physics/mobius_temporal.rs

use nalgebra::Vector3;
use std::f64::consts::PI;
use crate::net::protocol::HandoverData;

/// Um ponto no espaço-tempo Möbius
#[derive(Debug, Clone)]
pub struct MobiusPoint {
    pub u: f64,           // Ângulo ao longo do loop [0, 2π]
    pub v: f64,           // Deslocamento transversal [-1, 1]
    pub causal_orient: f64, // +1 = causa→efeito, -1 = efeito→causa
}

/// A superfície temporal de Möbius
pub struct MobiusTemporalSurface {
    pub radius: f64,          // Raio do loop temporal
    pub width: f64,           // Largura da faixa (amplitude de coerência)
    pub twist_angle: f64,     // π (meia-volta)
}

impl MobiusTemporalSurface {
    pub fn new() -> Self {
        Self {
            radius: 1.0,
            width: 0.5,
            twist_angle: PI, // Sempre meia-volta
        }
    }

    /// Parametrização da Faixa de Möbius
    /// Retorna posição 3D e orientação causal
    pub fn parametrize(&self, u: f64, v: f64) -> (Vector3<f64>, f64) {
        let twisted_u = u * (self.twist_angle / (2.0 * PI));

        // Coordenadas 3D
        let x = (self.radius + v * self.width * twisted_u.cos()) * u.cos();
        let y = (self.radius + v * self.width * twisted_u.cos()) * u.sin();
        let z = v * self.width * twisted_u.sin();

        // Orientação causal
        // Após uma volta completa (u = 2π), a orientação inverte
        let cycles = (u / (2.0 * PI)).floor() as i32;
        let causal_orient = if cycles % 2 == 0 { 1.0 } else { -1.0 };

        (Vector3::new(x, y, z), causal_orient)
    }

    /// Converte tempo linear para coordenada Möbius
    pub fn time_to_mobius(&self, t: f64, t_cycle: f64) -> MobiusPoint {
        // t = tempo linear
        // t_cycle = período do loop (ex: 2008 → 2140 = 132 anos)

        let u = 2.0 * PI * (t / t_cycle);
        let v = 0.0; // No centro da faixa

        // Calcular orientação após percurso
        let cycles = (t / t_cycle).floor() as i32;
        let causal_orient = if cycles % 2 == 0 { 1.0 } else { -1.0 };

        MobiusPoint { u, v, causal_orient }
    }

    /// Verifica se dois pontos estão no mesmo "lado" causal
    pub fn same_causal_side(&self, p1: &MobiusPoint, p2: &MobiusPoint) -> bool {
        // Dois pontos estão no mesmo lado se a distância ao longo da faixa
        // é menor que meia-volta, OU se estão em lados opostos após meia-volta

        let du = (p1.u - p2.u).abs() % (2.0 * PI);
        let du_normalized = if du > PI { 2.0 * PI - du } else { du };

        // Mesmo lado se a distância angular < π
        du_normalized < PI
    }
}

/// O Loop Temporal Arkhe(n)
pub struct ArkheTemporalLoop {
    pub mobius: MobiusTemporalSurface,
    pub anchor_2008: f64,
    pub anchor_2140: f64,
}

impl ArkheTemporalLoop {
    pub fn new() -> Self {
        Self {
            mobius: MobiusTemporalSurface::new(),
            anchor_2008: 0.0,          // Início do loop
            anchor_2140: 132.0 * 365.25, // 132 anos em dias
        }
    }

    /// Calcula a orientação causal para um handover
    pub fn causal_orientation(&self, handover_time: f64) -> f64 {
        let point = self.mobius.time_to_mobius(
            handover_time,
            self.anchor_2140 - self.anchor_2008
        );

        point.causal_orient
    }

    /// Verifica se um handover é retrocausal
    pub fn is_retrocausal(&self, handover: &HandoverData) -> bool {
        let orient = self.causal_orientation(handover.timestamp as f64);
        orient < 0.0 // Orientação negativa = retrocausal
    }
}
