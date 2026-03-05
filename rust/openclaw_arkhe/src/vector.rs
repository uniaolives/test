// rust/openclaw_arkhe/src/vector.rs
// Implementation of 6D OpenClaw Katharos Vector (Ω+224)

use nalgebra::Vector6;
use serde::{Deserialize, Serialize};

/// 6D OpenClaw Katharos Vector (Ω+224)
/// Components: [Bio, Aff, Soc, Cog, π, V]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct OpenClawKatharosVector {
    pub components: Vector6<f64>,
    pub weights: Vector6<f64>,
}

impl OpenClawKatharosVector {
    pub fn new(bio: f64, aff: f64, soc: f64, cog: f64, policy_entropy: f64, value_fn: f64) -> Self {
        Self {
            components: Vector6::new(bio, aff, soc, cog, policy_entropy, value_fn),
            weights: Vector6::new(0.35, 0.30, 0.20, 0.15, 0.0, 0.0), // Classic weights for first 4
        }
    }

    /// Extract classic 4D VK
    pub fn to_classic_vk(&self) -> [f64; 4] {
        [
            self.components[0],
            self.components[1],
            self.components[2],
            self.components[3],
        ]
    }

    /// Projection into the first 4 components
    pub fn project_classic(&self) -> nalgebra::Vector4<f64> {
        self.components.fixed_rows::<4>(0).into_owned()
    }
}

impl Default for OpenClawKatharosVector {
    fn default() -> Self {
        Self {
            components: Vector6::zeros(),
            weights: Vector6::new(0.35, 0.30, 0.20, 0.15, 0.0, 0.0),
        }
    }
}
