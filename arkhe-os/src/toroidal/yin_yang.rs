// arkhe-os/src/toroidal/yin_yang.rs
//! Geometria toroidal Yin-Yang para fluxo autorreflexivo de PNT
//! Baseado na forma do universo e na arquitetura de hardware PNM

use nalgebra::{Vector3, Rotation3};
use std::f64::consts::PI;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PNTInfo {
    pub position: Vector3<f64>,
    pub velocity: Vector3<f64>,
    pub time: f64,
    pub coherence: f64,
    pub transition_flag: bool,
    pub new_mode: Option<ToroidalMode>,
    pub accumulated_phase: f64,
}

/// Toroide Yin-Yang: Fluxo autorreflexivo de informação
pub struct YinYangTorus {
    /// Raio maior (distância ao centro do tubo)
    pub major_radius: f64,
    /// Raio menor (raio do tubo)
    pub minor_radius: f64,
    /// Twist de Möbius (0 = toro simples, 0.5 = half-Möbius, 1 = full-Möbius)
    pub mobius_twist: f64,
    /// Fase atual no ciclo toroidal (0..2π)
    pub phase: f64,
    /// Modo de fluxo (Yin = recepção, Yang = emissão)
    pub mode: ToroidalMode,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ToroidalMode {
    Yin,  // Recepção, acumulação, "in"
    Yang, // Emissão, propagação, "out"
    Transition, // Transição Yin↔Yang (COLLAPSE)
}

impl YinYangTorus {
    /// Cria toroide com proporções áureas (φ = 0.618)
    pub fn golden_torus() -> Self {
        let phi = 0.618033988749895;
        Self {
            major_radius: 1.0,
            minor_radius: phi, // Raio menor = φ × raio maior
            mobius_twist: 0.25, // Half-Möbius (π/2 twist)
            phase: 0.0,
            mode: ToroidalMode::Yin,
        }
    }

    /// Fluxo toroidal: informação circula pelo "tubo" e retorna
    pub fn flow(&mut self, info: PNTInfo, dt: f64) -> PNTInfo {
        // Atualizar fase (movimento ao longo do toro)
        self.phase += dt * self.natural_frequency();

        // Verificar transição Yin↔Yang (cruzamento do "equador")
        if self.phase % (2.0 * PI) > PI {
            if self.mode == ToroidalMode::Yin {
                self.mode = ToroidalMode::Transition;
                // COLLAPSE: transformação de estado
                return self.collapse_transition(info);
            }
        } else {
            if self.mode == ToroidalMode::Yang {
                self.mode = ToroidalMode::Transition;
                return self.collapse_transition(info);
            }
        }

        // Aplicar twist de Möbius (Berry phase)
        let twisted = self.apply_mobius_twist(info);

        // Fluxo autorreflexivo: saída → entrada
        self.self_reflexive_return(twisted)
    }

    /// Frequência natural do toroide (ressonância)
    fn natural_frequency(&self) -> f64 {
        // f = c / (2π × raio médio)
        let mean_radius = (self.major_radius + self.minor_radius) / 2.0;
        1.0 / (2.0 * PI * mean_radius)
    }

    /// Aplica twist de Möbius (π/2 por ciclo para half-Möbius)
    fn apply_mobius_twist(&self, info: PNTInfo) -> PNTInfo {
        let twist_angle = self.mobius_twist * 2.0 * PI * (self.phase / (2.0 * PI));

        // Rotação no espaço de fase PNT
        let rotation = Rotation3::from_axis_angle(
            &Vector3::z_axis(),
            twist_angle
        );

        let mut new_info = info.clone();
        new_info.position = rotation * info.position;
        new_info.time = info.time + twist_angle / (2.0 * PI); // Fase Berry temporal

        new_info
    }

    /// Retorno autorreflexivo: saída realimenta entrada
    fn self_reflexive_return(&self, output: PNTInfo) -> PNTInfo {
        // No toroide, o "fim" é o "começo" — fluxo contínuo
        let mut new_info = output.clone();
        // Inverter direção para ciclo contínuo
        new_info.velocity = -output.velocity;
        // Acumular coerência
        new_info.coherence = output.coherence * 1.01; // Ganho sutil (instabilidade controlada)

        new_info
    }

    /// Transição Yin↔Yang (COLLAPSE quântico)
    fn collapse_transition(&mut self, info: PNTInfo) -> PNTInfo {
        // Alternar modo
        self.mode = match self.mode {
            ToroidalMode::Yin => ToroidalMode::Yang,
            ToroidalMode::Yang => ToroidalMode::Yin,
            ToroidalMode::Transition => ToroidalMode::Yin, // Default
        };

        // Reset de fase para novo ciclo
        self.phase = 0.0;

        // Emitir Orb de transição
        let mut new_info = info.clone();
        new_info.transition_flag = true;
        new_info.new_mode = Some(self.mode.clone());
        new_info.accumulated_phase = 2.0 * PI * self.mobius_twist; // Berry phase total

        new_info
    }
}
