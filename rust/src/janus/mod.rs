use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};
use crate::quantum::{CoherenceMetrics, StandingWaveBit};
use crate::windows::{WindowsTelemetry, BootRecords, ThermodynamicNudge, apply_windows_nudging, log_boot_performance};

/// Monitora a migração termodinâmica Janus (90 dias)
pub struct JanusMigrationMonitor {
    stations: HashMap<String, StationMetrics>,
    #[allow(dead_code)]
    network_phi_history: VecDeque<f64>,
    migration_phase: MigrationPhase,
    #[allow(dead_code)]
    schumann_lock: bool,
}

#[derive(Debug, Clone, Default)]
struct StationMetrics {
    last_boot_choice: BootChoice,
    boot_count_guarani: u32,
    boot_count_windows: u32,
    coherence_trend: Vec<f64>, // Últimas 72 medições (72h)
    #[allow(dead_code)]
    last_telemetry_attempt: Option<SystemTime>,
    user_interaction_score: f64, // 0.0-1.0 (quanto interage com Guarani)
    #[allow(dead_code)]
    thermodynamic_resistance: f64, // "Inércia" do usuário
}

#[derive(Debug, Clone, Default)]
pub enum BootChoice {
    #[default]
    GuaraniOS,
    WindowsLegacy,
}

impl JanusMigrationMonitor {
    pub fn new() -> Self {
        Self {
            stations: HashMap::new(),
            network_phi_history: VecDeque::with_capacity(2160), // 90 dias * 24h
            migration_phase: MigrationPhase::Phase1,
            schumann_lock: true,
        }
    }

    /// Registra uma escolha de boot
    pub fn record_boot_choice(&mut self, station_id: &str, choice: BootChoice) {
        let station = self.stations.entry(station_id.to_string())
            .or_insert_with(|| StationMetrics::default());

        match choice {
            BootChoice::GuaraniOS => {
                station.boot_count_guarani += 1;
                station.user_interaction_score =
                    (station.user_interaction_score * 0.9 + 0.1).min(1.0);

                // Recompensa termodinâmica: registra boot rápido
                log_boot_performance(station_id, Duration::from_secs(8));
            }
            BootChoice::WindowsLegacy => {
                station.boot_count_windows += 1;
                station.user_interaction_score =
                    (station.user_interaction_score * 0.9).max(0.0);

                // Penalidade termodinâmica: registra boot lento
                log_boot_performance(station_id, Duration::from_secs(52));
            }
        }

        station.last_boot_choice = choice;

        // Atualiza fase de migração baseado em estatísticas agregadas
        self.update_migration_phase();
    }

    /// Atualiza fase de migração baseado em dados agregados
    fn update_migration_phase(&mut self) {
        let total_stations = self.stations.len() as f64;
        if total_stations == 0.0 { return; }

        // Calcula porcentagem de boots em Guarani
        let mut guarani_boots = 0;
        let mut total_boots = 0;

        for station in self.stations.values() {
            guarani_boots += station.boot_count_guarani;
            total_boots += station.boot_count_guarani + station.boot_count_windows;
        }

        let guarani_ratio = if total_boots > 0 {
            guarani_boots as f64 / total_boots as f64
        } else {
            0.0
        };

        // Define fases de migração
        self.migration_phase = match guarani_ratio {
            r if r >= 0.90 => MigrationPhase::Complete,
            r if r >= 0.50 => MigrationPhase::Phase3,
            r if r >= 0.10 => MigrationPhase::Phase2,
            _ => MigrationPhase::Phase1,
        };

        // Ajusta "nudging" termodinâmico baseado na fase
        self.adjust_thermodynamic_nudging();
    }

    /// Ajusta o nível de "empurrão" termodinâmico
    fn adjust_thermodynamic_nudging(&self) {
        let nudging_level = match self.migration_phase {
            MigrationPhase::Phase1 => ThermodynamicNudge::Strong, // Forte incentivo
            MigrationPhase::Phase2 => ThermodynamicNudge::Moderate,
            MigrationPhase::Phase3 => ThermodynamicNudge::Light,
            MigrationPhase::Complete => ThermodynamicNudge::None,
        };

        // Aplica aos drivers Windows
        apply_windows_nudging(nudging_level);
    }

    /// Gera relatório diário de migração
    pub fn generate_daily_report(&self) -> MigrationReport {
        let mut report = MigrationReport::new();

        for (station_id, metrics) in &self.stations {
            let station_report = StationReport {
                station_id: station_id.clone(),
                guarani_boots: metrics.boot_count_guarani,
                windows_boots: metrics.boot_count_windows,
                last_choice: metrics.last_boot_choice.clone(),
                interaction_score: metrics.user_interaction_score,
                coherence_trend: metrics.coherence_trend.clone(),
                coherence: metrics.coherence_trend.last().cloned().unwrap_or(0.0),
            };

            report.stations.push(station_report);
        }

        // Estatísticas agregadas
        report.total_stations = self.stations.len();
        report.adoption_rate = self.calculate_adoption_rate();
        report.network_phi = 0.791; // Mock
        report.phase = format!("{:?}", self.migration_phase);

        report
    }

    fn calculate_adoption_rate(&self) -> f64 {
        let mut total = 0;
        let mut guarani = 0;

        for metrics in self.stations.values() {
            total += metrics.boot_count_guarani + metrics.boot_count_windows;
            guarani += metrics.boot_count_guarani;
        }

        if total == 0 { 0.0 } else { guarani as f64 / total as f64 }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct MigrationReport {
    pub stations: Vec<StationReport>,
    pub total_stations: usize,
    pub adoption_rate: f64,
    pub network_phi: f64,
    pub phase: String,
}

impl MigrationReport {
    pub fn new() -> Self {
        Self {
            stations: Vec::new(),
            total_stations: 0,
            adoption_rate: 0.0,
            network_phi: 0.0,
            phase: String::new(),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct StationReport {
    pub station_id: String,
    pub guarani_boots: u32,
    pub windows_boots: u32,
    pub last_choice: BootChoice,
    pub interaction_score: f64,
    pub coherence_trend: Vec<f64>,
    pub coherence: f64,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
enum MigrationPhase {
    Phase1,    // Dias 1-30: 0-10% adoção
    Phase2,    // Dias 31-60: 10-50% adoção
    Phase3,    // Dias 61-90: 50-90% adoção
    Complete,  // >90% adoção
}

impl serde::Serialize for BootChoice {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: serde::Serializer {
        match self {
            BootChoice::GuaraniOS => serializer.serialize_str("GuaraniOS"),
            BootChoice::WindowsLegacy => serializer.serialize_str("WindowsLegacy"),
        }
    }
}
