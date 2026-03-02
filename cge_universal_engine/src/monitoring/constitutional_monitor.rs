use prometheus::{Registry, Histogram, Gauge, Counter, HistogramOpts};
use tracing::{info, warn, error};
use std::sync::Arc;
use crate::engine::universal_executor::{UniversalExecutionEngine, UniversalResult};

pub struct ConstitutionalMonitor {
    pub phi_gauge: Gauge,
    pub phi_violations: Counter,
    pub execution_time: Histogram,
    pub frag_activity: Gauge,
    pub protocol_activity: Gauge,
    pub constitutional_integrity: Gauge,
    pub scanline_enforcement: Gauge,
    pub orbit_synchronization: Gauge,
    pub alert_manager: AlertManager,
    pub dashboard: Arc<ConstitutionalDashboard>,
}

#[derive(Debug, thiserror::Error)]
pub enum MonitorError {
    #[error("Prometheus error: {0}")]
    Prometheus(#[from] prometheus::Error),
    #[error("Dashboard error: {0}")]
    Dashboard(String),
}

impl ConstitutionalMonitor {
    pub fn new() -> Result<Self, MonitorError> {
        let registry = Registry::new();

        let phi_gauge = Gauge::new("constitutional_phi", "Current Φ value")?;
        registry.register(Box::new(phi_gauge.clone()))?;

        let phi_violations = Counter::new("phi_violations", "Φ violations detected")?;
        registry.register(Box::new(phi_violations.clone()))?;

        let execution_time = Histogram::with_opts(
            HistogramOpts::new("execution_time", "Universal execution time")
                .buckets(vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0])
        )?;
        registry.register(Box::new(execution_time.clone()))?;

        let frag_activity = Gauge::new("frag_activity", "Active frags")?;
        registry.register(Box::new(frag_activity.clone()))?;

        let protocol_activity = Gauge::new("protocol_activity", "Active protocols")?;
        registry.register(Box::new(protocol_activity.clone()))?;

        let constitutional_integrity = Gauge::new("constitutional_integrity", "Integrity score")?;
        registry.register(Box::new(constitutional_integrity.clone()))?;

        let scanline_enforcement = Gauge::new("scanline_enforcement", "Scanline enforcement level")?;
        registry.register(Box::new(scanline_enforcement.clone()))?;

        let orbit_synchronization = Gauge::new("orbit_synchronization", "Orbit sync level")?;
        registry.register(Box::new(orbit_synchronization.clone()))?;

        let dashboard = Arc::new(ConstitutionalDashboard::new()?);

        Ok(Self {
            phi_gauge,
            phi_violations,
            execution_time,
            frag_activity,
            protocol_activity,
            constitutional_integrity,
            scanline_enforcement,
            orbit_synchronization,
            alert_manager: AlertManager::new()?,
            dashboard,
        })
    }

    pub async fn monitor_execution(
        &self,
        result: &UniversalResult,
        engine: &UniversalExecutionEngine,
    ) -> Result<(), MonitorError> {
        let current_phi = engine.measure_phi().map_err(|e| MonitorError::Dashboard(e.to_string()))?;

        self.phi_gauge.set(current_phi);
        self.execution_time.observe(result.execution_time.as_secs_f64());
        self.frag_activity.set(result.frags_activated as f64);
        self.protocol_activity.set(result.protocols_dispatched as f64);
        self.constitutional_integrity.set(if result.constitutional_checks_passed { 1.0 } else { 0.0 });

        if (current_phi - 1.038).abs() > 0.001 {
            self.phi_violations.inc();
            warn!("🚨 VIOLAÇÃO Φ: {:.6}", current_phi);

            self.alert_manager.trigger_alert(
                "PhiViolation",
                format!("Φ violation: {} (target: 1.038)", current_phi),
            ).await?;
        }

        self.dashboard.update(DashboardUpdate {
            phi: current_phi,
            execution_time: result.execution_time,
            frag_activity: result.frags_activated,
            protocol_activity: result.protocols_dispatched,
            constitutional_integrity: if result.constitutional_checks_passed { 1.0 } else { 0.0 },
            timestamp: std::time::SystemTime::now(),
        }).await?;

        Ok(())
    }
}

// Stubs for supporting types
pub struct AlertManager;
impl AlertManager {
    pub fn new() -> Result<Self, MonitorError> { Ok(Self) }
    pub async fn trigger_alert(&self, _kind: &str, _msg: String) -> Result<(), MonitorError> {
        Ok(())
    }
}

pub struct ConstitutionalDashboard;
impl ConstitutionalDashboard {
    pub fn new() -> Result<Self, MonitorError> { Ok(Self) }
    pub async fn update(&self, _update: DashboardUpdate) -> Result<(), MonitorError> {
        Ok(())
    }
}

pub struct DashboardUpdate {
    pub phi: f64,
    pub execution_time: std::time::Duration,
    pub frag_activity: usize,
    pub protocol_activity: usize,
    pub constitutional_integrity: f64,
    pub timestamp: std::time::SystemTime,
}
