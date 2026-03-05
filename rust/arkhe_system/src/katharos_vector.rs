// rust/arkhe_system/src/katharos_vector.rs
// Implementation of Katharós Vector and Qualic Permeability (ACPS Blueprint)

use std::sync::Arc;

/// Vector Katharos n-dimensional (Eq. 1 Blueprint)
#[derive(Clone, Debug)]
pub struct KatharosVector {
    /// Dimensions: Bio(0.35), Aff(0.30), Soc(0.20), Cog(0.15)
    pub components: [f64; 4],
    pub weights: [f64; 4],
    /// Calibrated reference (KTP)
    pub vk_ref: Option<Arc<KatharosVector>>,
    /// Time in Katharos Range (t_KR)
    pub t_kr: f64,
}

impl Default for KatharosVector {
    fn default() -> Self {
        Self {
            components: [0.0; 4],
            weights: [0.35, 0.30, 0.20, 0.15],
            vk_ref: None,
            t_kr: 0.0,
        }
    }
}

/// Dual Neuroception: NP (subcortical) vs NM (cortical)
pub enum Neuroception {
    Primary([f64; 4]),   // <100ms, binary SIGUR/PERICOL
    Mature(Vec<f64>),    // 2-5s, contextual
}

/// CAMADA -1: GEMINI (Biological Timechain)
#[derive(Clone, Debug)]
pub struct GeminiLayer {
    pub timestamp: f64,
    pub intensity: f64,
    pub thickness: f64,
    pub spatial_coords: [f64; 3],
}

impl GeminiLayer {
    /// Map GEMINI spatial data to cluster permeability (Q_cluster)
    pub fn intensity_to_permeability(intensity: f64) -> f64 {
        // High intensity (inflammation/stress) -> Low Q
        // Sigmoid collapse at threshold 0.30
        1.0 / (1.0 + (10.0 * (intensity - 0.30)).exp())
    }
}

/// Qualic Permeability Q(t) ∈ [0,1]
pub struct QualicPermeability {
    pub p_eff: f64,
    pub delta_k: f64,
    pub c_neuro: f64,
    pub q: f64,
}

impl KatharosVector {
    /// Compute Homeostatic Deviation (Delta K)
    pub fn compute_delta_k(&self) -> f64 {
        if let Some(ref_vk) = &self.vk_ref {
            let mut sum_sq = 0.0;
            for i in 0..4 {
                let diff = self.components[i] - ref_vk.components[i];
                sum_sq += self.weights[i] * diff.powi(2);
            }
            sum_sq.sqrt()
        } else {
            0.0
        }
    }

    /// Bifurcação θ - transição Q≈1 → Q≈0 (Eq. 11, 11a)
    pub fn theta_bifurcation(&self, stress_level: f64) -> f64 {
        let theta_0 = 100.0;  // prag basal ontogenético
        let beta_theta = 0.5; // amplificação por inflação
        let i = self.neuroceptive_inflation();
        theta_0 * (1.0 + beta_theta * i) * (1.0 + stress_level)
    }

    fn neuroceptive_inflation(&self) -> f64 {
        // Mock: inflação baseada no desvio homeostático
        self.compute_delta_k() * 2.0
    }

    /// Split Vectorial Katharos (SVK) - Eq. 6
    pub fn compute_svk_dd(&self, v_np: &Neuroception, v_nm: &Neuroception) -> f64 {
        match (v_np, v_nm) {
            (Neuroception::Primary(np), Neuroception::Mature(nm)) => {
                let mut sum_sq = 0.0;
                for i in 0..4 {
                    let nm_val = nm.get(i).cloned().unwrap_or(0.0);
                    let diff = np[i] - nm_val;
                    sum_sq += (diff * self.weights[i]).powi(2);
                }
                sum_sq.sqrt()
            }
            _ => 0.0
        }
    }
}
