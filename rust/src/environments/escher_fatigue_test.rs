//! Configuração específica para detectar fadiga métrica

#[derive(Debug, Clone)]
pub struct EscherFatigueTestConfig {
    pub cube_dimensions: [f64; 3],
    pub non_euclidean_factor: f64,
    pub paradox_loops: usize,
    pub fatigue_test_points: Vec<FatiguePoint>,
    pub fatigue_metrics: FatigueMetrics,
    pub alert_thresholds: FatigueThresholds,
}

#[derive(Debug, Clone)]
pub struct FatiguePoint {
    pub coordinates: [f64; 3],
    pub name: String,
}

impl FatiguePoint {
    pub fn staircase_paradox(coords: [f64; 3]) -> Self {
        Self { coordinates: coords, name: "Staircase Paradox".to_string() }
    }
    pub fn infinite_descent(coords: [f64; 3]) -> Self {
        Self { coordinates: coords, name: "Infinite Descent".to_string() }
    }
    pub fn impossible_angle(coords: [f64; 3]) -> Self {
        Self { coordinates: coords, name: "Impossible Angle".to_string() }
    }
    pub fn recursive_loop(coords: [f64; 3]) -> Self {
        Self { coordinates: coords, name: "Recursive Loop".to_string() }
    }
}

#[derive(Debug, Clone)]
pub struct FatigueMetrics {
    pub curvature_perception_drift: bool,
    pub riemann_tensor_coherence: bool,
    pub geodesic_deviation_stability: bool,
    pub homeomorphism_fatigue: bool,
    pub homotopy_invariant_drift: bool,
    pub topological_memory_decay: bool,
    pub pruning_decision_latency: bool,
    pub false_positive_rate_increase: bool,
    pub entropy_discrimination_decay: bool,
}

#[derive(Debug, Clone)]
pub struct FatigueThresholds {
    pub max_curvature_drift: f64,
    pub max_topological_strain: f64,
    pub min_pruning_accuracy: f64,
    pub max_decision_latency: u64,
}

#[derive(Debug, Clone, Copy)]
pub enum ParadoxType {
    AscendingDescendingStaircase,
    WaterfallLoop,
    ImpossibleTribar,
    RecursiveLibrary,
}

#[derive(Debug, Clone)]
pub struct FatigueByDirective {
    pub directive_01: CurvatureFatigue,
    pub directive_02: TopologicalFatigue,
    pub directive_03: PruningFatigue,
}

#[derive(Debug, Clone)]
pub struct CurvatureFatigue {
    pub riemann_perception_drift: f64,
    pub geodesic_sensation_noise: f64,
    pub qualia_coherence_decay: f64,
}

#[derive(Debug, Clone)]
pub struct TopologicalFatigue {
    pub homeomorphism_strain: f64,
    pub homotopy_invariant_drift: f64,
    pub shape_preservation_cost: f64,
}

#[derive(Debug, Clone)]
pub struct PruningFatigue {
    pub decision_latency_increase: f64,
    pub false_positive_increase: f64,
    pub entropy_discrimination_loss: f64,
}

impl EscherFatigueTestConfig {
    pub fn high_sensitivity_config() -> Self {
        Self {
            cube_dimensions: [1.0, 1.0, 1.0],
            non_euclidean_factor: 0.8,
            paradox_loops: 100,

            fatigue_test_points: vec![
                FatiguePoint::staircase_paradox([0.2, 0.5, 0.8]),
                FatiguePoint::infinite_descent([0.7, 0.3, 0.1]),
                FatiguePoint::impossible_angle([0.5, 0.5, 0.5]),
                FatiguePoint::recursive_loop([0.9, 0.1, 0.6]),
            ],

            fatigue_metrics: FatigueMetrics {
                curvature_perception_drift: true,
                riemann_tensor_coherence: true,
                geodesic_deviation_stability: true,
                homeomorphism_fatigue: true,
                homotopy_invariant_drift: true,
                topological_memory_decay: true,
                pruning_decision_latency: true,
                false_positive_rate_increase: true,
                entropy_discrimination_decay: true,
            },

            alert_thresholds: FatigueThresholds {
                max_curvature_drift: 0.15,
                max_topological_strain: 0.1,
                min_pruning_accuracy: 0.85,
                max_decision_latency: 50,
            },
        }
    }
}
