// src/main.rs
use cge_constitutional_system::{ConstitutionalSystem, CGE_VERSION};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏛️ CGE Constitutional System {}", CGE_VERSION);
    let mut system = ConstitutionalSystem::new();

    let mut params = HashMap::new();
    params.insert("level".to_string(), "strict".to_string());

    let receipt = system.execute_constitutional_operation(
        "asi_enable",
        &params,
        "did:plc:arquiteto-omega"
    ).await?;

    println!("✅ Operação constitucional completada!");
    println!("Recibo ID: {:?}", receipt.operation_id);
    println!("Bloco CGE: #{}", receipt.cge_block_number);

    Ok(())
}
