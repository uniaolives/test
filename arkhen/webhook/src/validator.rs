use crate::crds::*;
use kube::{Client, Api, Resource, ResourceExt};
use kube::core::admission::{AdmissionRequest, AdmissionResponse};
use serde::Serialize;
use serde_json::Value;

pub fn validate_node_basic(spec: &QuantumManifoldNodeSpec) -> Result<(), String> {
    if spec.desired_phi < 0.0 || spec.desired_phi > 1.0 {
        return Err("desiredPhi must be between 0 and 1".into());
    }
    if let Some(spin) = spec.spin {
        if spin < -1.0 || spin > 1.0 {
             return Err("spin must be between -1 and 1".into());
        }
    }
    match spec.node_type.as_str() {
        "satellite" | "groundStation" | "fusionCenter" | "sensorArray" => Ok(()),
        _ => Err(format!("Unknown nodeType: {}", spec.node_type)),
    }
}

pub async fn validate_spin_conservation(
    new_node_spec: &QuantumManifoldNodeSpec,
    client: &Client,
    namespace: &str,
) -> Result<(), String> {
    let nodes_api: Api<QuantumManifoldNode> = Api::namespaced(client.clone(), namespace);
    let nodes = nodes_api.list(&Default::default()).await
        .map_err(|e| format!("Failed to list nodes: {}", e))?;

    let current_spin_sum: f64 = nodes.items.iter()
        .map(|n| n.spec.spin.unwrap_or(0.0))
        .sum();
    let new_spin = new_node_spec.spin.unwrap_or(0.0);
    let new_sum = current_spin_sum + new_spin;

    const MAX_SPIN: f64 = 100.0;
    if new_sum.abs() > MAX_SPIN {
        return Err(format!("Total spin would exceed limits: |{:.2}| > {:.2}", new_sum, MAX_SPIN));
    }
    Ok(())
}

pub async fn validate_admission_review<T: Resource + Serialize>(
    req: &AdmissionRequest<T>,
    client: &Client,
) -> AdmissionResponse {
    let mut resp = AdmissionResponse::from(req);
    let namespace = req.namespace.as_deref().unwrap_or("default");

    let obj = match &req.object {
        Some(o) => o,
        None => return resp.deny("No object provided"),
    };

    let kind = req.kind.kind.as_str();
    let group = req.kind.group.as_str();

    // Fix: extract 'spec' from the full object
    let full_obj = serde_json::to_value(obj).unwrap();
    let spec_obj = full_obj.get("spec").cloned().unwrap_or(Value::Null);

    let validation_result = match (group, kind) {
        ("arkhe.quantum", "QuantumManifoldNode") => {
            match serde_json::from_value::<QuantumManifoldNodeSpec>(spec_obj) {
                Ok(spec) => {
                    if let Err(e) = validate_node_basic(&spec) {
                        Err(e)
                    } else {
                        validate_spin_conservation(&spec, client, namespace).await
                    }
                }
                Err(e) => Err(format!("Failed to parse QuantumManifoldNode spec: {}", e)),
            }
        }
        ("arkhe.quantum", "QuantumChannel") => {
            Ok(())
        }
        _ => Ok(()),
    };

    match validation_result {
        Ok(_) => {
            resp.allowed = true;
            resp
        }
        Err(msg) => resp.deny(msg),
    }
}
