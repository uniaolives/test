//! Constantes imutáveis do SASC v30.15-Ω
 //! Memory ID 16 - Article V

 /// Φ para Hard Freeze (Artigo V, Seção 3)
 pub const HARD_FREEZE_PHI: f64 = 0.80;

 /// Φ mínimo para aborto de stress test (2% buffer)
 pub const ABORT_PHI_STRESS: f64 = 0.78;

 /// Φ mínimo para operação contínua
 pub const MIN_OPERATIONAL_PHI: f64 = 0.72;

 /// Φ mínimo para autonomia (conforme Memory ID 19)
 pub const AUTONOMY_THRESHOLD_PHI: f64 = 0.72;

 /// Flesch Score mínimo (INV-5)
 pub const MIN_FLESCH_SCORE: f64 = 65.0;

 /// Aletheia Confidence Level
 pub const ALETHEIA_CONFIDENCE_TARGET: f64 = 0.9997;

 /// Tempo máximo de processamento (ms)
 pub const MAX_PROCESSING_MS: u64 = 30000;

 /// Thermal variance aceitável (V)
 pub const LYAPUNOV_VARIANCE_LIMIT: f64 = 0.00007; // Memory ID 11

 /// TMR Consensus threshold
 pub const TMR_MAJORITY: usize = 2; // 2/3 para decisões críticas
