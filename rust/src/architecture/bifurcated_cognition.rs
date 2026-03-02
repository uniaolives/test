use std::collections::HashMap;
use std::time::Duration;
use crate::crystallization::types::{SkillSignature, NativeModule};
use crate::philosophy::ennead_framework::EnneadCore;
use crate::crystallization::guardrail_engine::CrystallizedGuardrail;

pub struct QuantumPhilosophicalCouncil {
    pub core: EnneadCore,
}

pub struct FluidThinking {
    pub council: QuantumPhilosophicalCouncil,
    pub entropy_budget: f64,
    pub allowed_duration: Duration,
}

pub struct CrystallizedExecution {
    pub skill_library: HashMap<SkillSignature, NativeModule>,
    pub guardrail_checker: CrystallizedGuardrail,
    pub fallback_threshold: f64,
}

pub struct CognitiveRouter {
    // Decision logic between Fluid and Crystallized
}

pub enum CognitiveMode {
    /// Modo Fluido (Sistema 2): Processamento pelo Conselho Axial completo
    /// Uso: Problemas novos, ambíguos, éticos complexos. Alto custo, alta flexibilidade.
    FluidThinking {
        council: QuantumPhilosophicalCouncil,
        entropy_budget: f64, // Alto
        allowed_duration: Duration, // Segundos
    },

    /// Modo Cristalizado (Sistema 1): Execução de módulos estáticos recompilados
    /// Uso: Tarefas reconhecidas, padrões estabelecidos. Baixíssimo custo, alta velocidade.
    CrystallizedExecution {
        skill_library: HashMap<SkillSignature, NativeModule>,
        guardrail_checker: CrystallizedGuardrail,
        fallback_threshold: f64, // Se incerteza > este valor, volta para Fluido
    },
}

pub struct BifurcatedAGI {
    pub fluid_core: FluidThinking,
    pub crystal_cache: CrystallizedExecution,
    pub mode_router: CognitiveRouter,
}
