//! Interface de linha de comando para o arkhe-os.
//! Permite criar tarefas e executar ciclos de escalonamento.

mod compiler;
mod kernel;
mod physics;
mod db;
mod net;
mod phys;

use kernel::scheduler::CoherenceScheduler;
use kernel::task::Task;
use compiler::parser::parse_intention_block;
use db::ledger::{TeknetLedger, HandoverRecord};
use phys::ibm_client::QuantumAntenna;
use net::{P2PNode, HandoverData};
use std::sync::Arc;
use tokio::time::{sleep, Duration};
use chrono::Utc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    println!("======================================================");
    println!(" 🜁 ArkheOS Shell v0.5 — Multi-Dimensional Interface");
    println!(" [!] Timechain (RocksDB Placeholder) conectada.");
    println!(" [!] Antena Horizontal (P2P) e Vertical (QPU) ativas.");
    println!("======================================================\n");

    // 1. Inicializa ou restaura o banco de dados
    let ledger = Arc::new(TeknetLedger::new("./arkhe_chain_db").expect("Erro fatal: DB não pôde ser montado."));
    let (restored_phi_q, restored_last_id) = ledger.restore_vacuum_state();

    let mut kernel = CoherenceScheduler::new(restored_phi_q);
    let mut task_id_counter = restored_last_id + 1;

    println!("[SYS] Boot concluído. Vácuo restaurado em φ_q = {:.3}\n", restored_phi_q);

    // 2. Iniciar Antena Horizontal (P2P)
    let p2p_node = P2PNode::new(7000, ledger.clone());
    tokio::spawn(async move {
        p2p_node.run_server().await;
    });

    // 3. Iniciar Antena Vertical (IBM Quantum) em background
    let antenna = QuantumAntenna::new("SEU_IBM_QUANTUM_TOKEN".to_string());
    tokio::spawn(async move {
        loop {
            sleep(Duration::from_secs(30)).await;
            if let Ok(real_phi_q) = antenna.measure_vacuum_quality("ibm_brisbane").await {
                println!("\n[SYSTEM] Vácuo recalibrado via hardware físico: φ_q = {:.3}", real_phi_q);
            }
        }
    });

    // 4. Simulação de Intenções (Exemplo de intention-lang)
    let raw_code = r#"
        intention stabilize_bitcoin {
            target: "2009-01-03"
            coherence: 0.9
            priority: high
            payload: "seed_genesis_block"
        }
    "#;

    match parse_intention_block(raw_code.trim()) {
        Ok((_, ast)) => {
            println!("[COMPILADOR] Intenção compilada: {}", ast.name);
            let task = Task::new(task_id_counter, &ast.name, ast.coherence, 5, ast.priority);
            kernel.schedule(task);

            // Simular execução imediata para demonstração de persistência/rede
            if let Some(event) = kernel.tick() {
                match event {
                    kernel::scheduler::SchedulerEvent::TaskStarted(t) => println!("[KERNEL] Task {} iniciada", t.id),
                    _ => {}
                }

                // Avançar ticks para concluir
                for _ in 0..5 {
                    if let Some(event) = kernel.tick() {
                        if let kernel::scheduler::SchedulerEvent::TaskCompleted(t) = event {
                             println!("[KERNEL] Task {} concluída", t.id);

                             let record = HandoverRecord {
                                id: t.id,
                                timestamp: Utc::now(),
                                intention_name: ast.name.clone(),
                                coherence_delta: ast.coherence,
                                phi_q_after: kernel.status().0,
                            };

                            ledger.commit_handover(&record).unwrap();
                            ledger.save_vacuum_state(kernel.status().0, t.id).unwrap();

                            // Broadcast para a rede
                            let _ = P2PNode::broadcast_handover("127.0.0.1:7001", HandoverData {
                                id: t.id,
                                timestamp: record.timestamp.timestamp(),
                                description: t.name,
                                phi_q_after: record.phi_q_after,
                            }).await;
                        }
                    }
                }
            }
            task_id_counter += 1;
        },
        Err(e) => println!("[ERRO] Falha ao compilar intenção: {:?}", e),
    }

    println!("\n[SYS] Ciclos iniciais concluídos. Mantendo antenas ativas...");

    // Manter vivo para as antenas e conexões P2P
    loop {
        sleep(Duration::from_secs(1)).await;
    }
}
