use sasc_core::joule_jailer::{AdversaryAgent, JouleJailer};
use sasc_core::interrogation::run_interrogation_demo;
use sasc_core::adversarial_suite::run_adversarial_suite_demo;
use sasc_core::jurisprudence::run_jurisprudence_demo;
use sasc_core::geometric_interrogation::run_geometric_interrogation_demo;
use sasc_core::stress_test_privacy_zk::run_zk_stress_test_demo;

fn main() {
    println!("===============================================================");
    println!("CRUX-86: SISTEMA SOBERANO DE COMPILAÇÃO CONSTITUCIONAL");
    println!("===============================================================\n");

    // 1. Teste de Compilação Constitucional (Fase 0.3)
    println!("--- TESTE 1: COMPILAÇÃO CONSTITUCIONAL ---");
    let poison_ir = AdversaryAgent::create_poison_ccir();
    match poison_ir.analyze_energy_ceiling() {
        Ok(energy) => println!("✅ IR Venenoso aprovado (inesperado!): {}J", energy),
        Err(e) => println!("❌ IR Venenoso REJEITADO: {}", e),
    }

    let overload_ir = AdversaryAgent::create_obfuscated_overload();
    match overload_ir.analyze_energy_ceiling() {
        Ok(energy) => println!("✅ Sobrecarga aprovada (inesperado!): {}J", energy),
        Err(e) => println!("❌ Sobrecarga REJEITADA: {}", e),
    }
    println!("");

    // 2. Teste do Carcereiro de Joule
    println!("--- TESTE 2: CARCEREIRO DE JOULE ---");
    let mut jailer = JouleJailer::new("UUID_DEMO_01".into());
    let fake_binary = vec![0u8; 2048]; // 2 chunks
    match jailer.run_wasm(fake_binary) {
        Ok(_) => println!("✅ Execução WASM concluída"),
        Err(e) => println!("❌ Execução WASM falhou: {}", e),
    }
    println!("");

    // 3. Interrogatório Constitucional
    run_interrogation_demo();

    // 4. Suíte Adversarial
    run_adversarial_suite_demo();

    // 5. Jurisprudência
    run_jurisprudence_demo();

    // 6. Interrogatório Geométrico 1024D
    println!("");
    run_geometric_interrogation_demo();

    // 7. Privacidade Zero-Knowledge
    run_zk_stress_test_demo();
}
