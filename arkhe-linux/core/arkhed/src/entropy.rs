use serde::{Serialize, Deserialize};

/// Unidade de Entropia Arkhe (AEU) – representada como f32.
pub type ArkheEntropyUnit = f32;

/// Constante de Planck efetiva do sistema (ajustável).
/// Define a escala da relação incerteza energia-tempo.
pub const ARKHE_PLANCK: f32 = 1.0; // unidades arbitrárias

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntropyStats {
    pub total_aeu: ArkheEntropyUnit,
    pub temperature: f32, // Effective system temperature (e.g. 1/IPC)
    pub von_neumann_entropy: f32,
}

pub struct EntropyMonitor;

impl EntropyMonitor {
    pub async fn attach_probes() -> Result<Self, anyhow::Error> {
        Ok(Self)
    }

    pub async fn collect(&self) -> EntropyStats {
        // Mocked implementation for now
        EntropyStats {
            total_aeu: 0.1,
            temperature: 1.0,
            von_neumann_entropy: 0.5,
        }
    }
}

/// Calcula o half-life de um handover baseado no custo de entropia.
///
/// # Argumentos
/// * `entropy_cost` - Custo em AEU.
/// * `system_temperature` - Temperatura efetiva do sistema (adimensional).
///
/// Retorna o half-life em milissegundos.
pub fn half_life_from_entropy(entropy_cost: ArkheEntropyUnit, system_temperature: f32) -> f32 {
    // Modelo: half-life ∝ 1 / (entropy_cost * T)
    // O fator de escala pode ser ajustado empiricamente.
    const KB: f32 = 1.0; // constante de Boltzmann normalizada
    let energy = entropy_cost * KB * system_temperature;
    if energy <= 0.0 {
        f32::INFINITY
    } else {
        ARKHE_PLANCK / (2.0 * energy)
    }
}

/// Probabilidade de sobrevivência após um tempo de trânsito.
pub fn survival_probability(half_life_ms: f32, transit_time_ms: f32) -> f32 {
    if half_life_ms <= 0.0 {
        1.0
    } else {
        (-transit_time_ms / half_life_ms).exp()
    }
}
