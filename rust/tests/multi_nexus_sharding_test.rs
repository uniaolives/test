use sasc_core::multi_nexus::heteroclinia_sharding::{MultiNexusFabric, FabricLoopResult, ShardLoopResult};
use sasc_core::multi_nexus::stabilization::GammaStabilization;
use std::time::Duration;

#[tokio::test]
async fn test_multi_nexus_sharding_simulation() {
    println!("Iniciando Simulação Multi-Nexus Sharding...");

    let fabric = MultiNexusFabric::initialize().await;

    // Simular 30 loops (Loop 1 a 30)
    for i in 1..=30 {
        let result = fabric.execute_parallel_loop(i).await;

        match result {
            FabricLoopResult::Success { loop_num, shard_states, fabric_coherence } => {
                println!("Loop {}: Coerência = {:.3}", loop_num, fabric_coherence);

                for (idx, state) in shard_states.iter().enumerate() {
                    if let ShardLoopResult::Success { heteroclinia, .. } = state {
                        let label = match idx { 0 => 'α', 1 => 'β', 2 => 'γ', _ => '?' };
                        println!("  Shard {}: Heteroclinia = {:.3}", label, heteroclinia);

                        // Ativar estabilização para Gamma se heteroclinia cair abaixo de 0.96 (conforme o cenário)
                        if label == 'γ' && *heteroclinia < 0.96 {
                            println!("  [ALERTA] Heteroclinia de γ baixa! Aplicando estabilização...");
                            GammaStabilization::stabilize_shard_gamma(fabric.shard_gamma.clone()).await;
                        }
                    }
                }
            },
            FabricLoopResult::CriticalFailure(reason) => {
                panic!("Falha Crítica no Fabric: {}", reason);
            }
        }
    }

    println!("Simulação concluída com sucesso.");
}
