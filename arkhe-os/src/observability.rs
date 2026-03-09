use metrics::{gauge, counter};

pub fn record_handover(success: bool, phi_q: f64, cost: f64) {
    let status = if success { "ok" } else { "fail" };
    counter!("arkhe_handover_total", 1, "status" => status);
    gauge!("arkhe_phi_q_gauge", phi_q);
    gauge!("arkhe_quantum_interest_cost", cost);
}
