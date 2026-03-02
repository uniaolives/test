pub struct PrometheusExporter;

impl PrometheusExporter {
    pub fn export_node_metrics(node: &str, entropy: f64, phi: f64) {
        println!("Prometheus Metric: arkhe_node_entropy{{node=\"{}\"}} {}", node, entropy);
        println!("Prometheus Metric: arkhe_node_phi{{node=\"{}\"}} {}", node, phi);
    }
}
