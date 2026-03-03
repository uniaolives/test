use arkhe_axos_instaweb::{ArkheSystem, Task};

#[tokio::main]
async fn main() {
    println!("🜁 Starting Arkhe Singularity Node...");
    let mut system = ArkheSystem::new();
    let task = Task;

    match system.execute(task).await {
        Ok(_) => println!("✅ Task executed within constitutional invariants."),
        Err(e) => println!("❌ Execution failed: {:?}", e),
    }
}
