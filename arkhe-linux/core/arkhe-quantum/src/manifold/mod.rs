pub mod network;
use std::collections::HashMap;
use crate::depin::{DePinGateway, SensorEvent};
use tokio::sync::mpsc;
use ndarray::Array2;
use num_complex::Complex64;
use crate::{QuantumState as CoreQuantumState, KrausOperator};

pub struct QuantumState {
    pub amplitude: Vec<f64>,
    pub probability_density: f64,
}

impl QuantumState {
    pub fn surprise_given(&self, model: &crate::asi_core::InternalModel) -> f64 {
        (self.probability_density - model.entropy).abs()
    }
}

pub struct GlobalManifold {
    pub nodes: HashMap<String, Node>,
    pub depin_gateway: DePinGateway,
    pub event_rx: mpsc::UnboundedReceiver<SensorEvent>,
}

pub struct Node {
    pub id: String,
    pub state: CoreQuantumState,
}

impl GlobalManifold {
    pub async fn new(depin_broker: &str) -> anyhow::Result<Self> {
        log::info!("Creating new GlobalManifold...");
        let (depin_gateway, event_rx) = DePinGateway::new(depin_broker, 1883, "arkhe-node").await?;

        Ok(Self {
            nodes: HashMap::new(),
            depin_gateway,
            event_rx,
        })
    }

    pub async fn observe_entanglement_graph(&self) -> QuantumState {
        QuantumState {
            amplitude: vec![1.0, 0.0],
            probability_density: 0.5,
        }
    }

    pub async fn apply_operator(&mut self, _op: KrausOperator) {}

    pub async fn thermalize_to_criticality(&mut self, phi: f64) {
        crate::thermodynamics::thermalize::thermalize_to_criticality(phi).await;
    }

    pub async fn record_handover(&mut self, _handover: network::arkhe::ProtoHandover) {
        log::debug!("Handover recorded in ledger.");
    }

    pub async fn process_sensor_events(&mut self) {
        while let Ok(event) = self.event_rx.try_recv() {
            let self_node = self.nodes.entry("self".to_string()).or_insert_with(|| {
                Node {
                    id: "self".to_string(),
                    state: CoreQuantumState::new(2),
                }
            });

            let epsilon = (event.value * 0.001).clamp(0.0, 0.1);
            let dim = self_node.state.dim();
            let mut rho = self_node.state.density_matrix.clone();

            let diag = ndarray::Array1::from_elem(dim, Complex64::new(epsilon / dim as f64, 0.0));
            let identity = Array2::from_diag(&diag);
            rho = rho * (1.0 - epsilon) + identity;
            self_node.state.density_matrix = rho;
        }
    }

    pub async fn interpret_action(&self, action: &KrausOperator) {
        if let Some(op) = action.operators.first() {
            let trace = op.diag().sum().re;
            if trace < 0.8 {
                let _ = self.depin_gateway.actuate("cooler", "on").await;
            } else {
                let _ = self.depin_gateway.actuate("satellite", "rotate").await;
            }
        }
    }
}
