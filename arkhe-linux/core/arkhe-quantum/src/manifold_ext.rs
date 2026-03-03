use arkhe_manifold::{GlobalManifold, QuantumState, Node};
use crate::ledger::OmegaLedger;
use crate::depin::{DePinGateway, SensorEvent};
use nalgebra::DMatrix;
use num_complex::Complex64;
use tokio::sync::mpsc;
use log::{info, debug};

pub struct ExtendedManifold {
    pub inner: GlobalManifold,
    pub ledger: OmegaLedger,
    pub depin_gateway: DePinGateway,
    pub event_rx: mpsc::UnboundedReceiver<SensorEvent>,
}

impl ExtendedManifold {
    pub async fn new(depin_broker: &str, ledger_path: &str) -> anyhow::Result<Self> {
        let ledger = OmegaLedger::open(ledger_path)?;
        let (gateway, rx) = DePinGateway::new_with_receiver(depin_broker, 1883, "arkhe-node").await?;

        Ok(Self {
            inner: GlobalManifold::new(),
            ledger,
            depin_gateway: gateway,
            event_rx: rx,
        })
    }

    pub async fn process_sensor_events(&mut self) {
        while let Ok(event) = self.event_rx.try_recv() {
            debug!("Sensor {}: valor = {}", event.sensor_id, event.value);

            let self_node = self.inner.nodes.entry("self".to_string()).or_insert_with(|| {
                let dim = 2;
                Node {
                    id: "self".to_string(),
                    state: QuantumState::maximally_mixed(dim),
                }
            });

            let epsilon = (event.value * 0.001).clamp(0.0, 0.1);
            let dim = self_node.state.dim();
            let mut rho = self_node.state.density_matrix.clone();

            let identity = DMatrix::from_diagonal_element(dim, dim, Complex64::new(epsilon / dim as f64, 0.0));
            rho = rho * Complex64::new(1.0 - epsilon, 0.0) + identity;

            let trace = rho.trace().re;
            if trace > 0.0 {
                rho /= Complex64::new(trace, 0.0);
            }

            self_node.state.density_matrix = rho;
            info!("Density matrix updated (entropy = {:.4})", self_node.state.von_neumann_entropy());
        }
    }

    pub async fn interpret_action(&self, action: &str) {
        if action.contains("dissipative") {
            let _ = self.depin_gateway.actuate("cooler", "on").await;
        } else {
            let _ = self.depin_gateway.actuate("satellite", "rotate").await;
        }
    }
}
