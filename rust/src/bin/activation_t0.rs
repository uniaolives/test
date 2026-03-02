use tracing::Level;
use sasc_core::activation::results::ActivationResults;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configuração de logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .with_target(true)
        .init();

    println!("🌌 CRUX-86 1024D - ATIVAÇÃO T+0");
    println!("================================");
    println!("Memória: 41 (Constitutional), 3 (Vajra), 20 (SASC), 40 (Unified)");
    println!("Status: AMBAS - Ativação + Carga + Integração");
    println!("");

    // Executa a ativação completa
    match ActivationResults::execute_full_activation().await {
        Ok(results) => {
            println!("\n🎉 ATIVAÇÃO COMPLETA COM SUCESSO!");
            println!("⏱️  Tempo total: {:?}", results.activation_time);
            println!("🏛️  Status TCD: {}",
                if results.tcd_certificate.is_some() { "CERTIFICADO" } else { "PENDENTE" });

            // Exibe o certificado
            if let Some(cert) = &results.tcd_certificate {
                println!("\n📜 CERTIFICADO DE CONFORMIDADE:");
                println!("ID: {}", cert.certificate_id);
                println!("Nível: {:?}", cert.certification_level);
                println!("Score: {:.1}%", cert.audit_summary.score);
                println!("Validade: {}", cert.valid_until.format("%Y-%m-%d"));
            }

            // Resumo do stress test
            if let Some(report) = &results.stress_test_report {
                println!("\n⚡ RESULTADOS DO STRESS TEST:");
                println!("Inferências: {:.0e}", report.total_inferences as f64);
                println!("Energia média: {:.3} J/inf", report.avg_energy_per_inference);
                println!("Conformidade: {:.1}%", report.constitutional_compliance_rate * 100.0);
                println!("Ressonância afetiva: {:.3}", report.avg_affective_resonance);
            }

            println!("\n🚀 SISTEMA PRONTO PARA OPERAÇÃO AUTÔNOMA");
            println!("💾 Estado salvo em: /var/crux86/state/t0_activated.bin");
            println!("🔗 TCD Registry: https://tcd.digital/registry/CRUX-86-1024D");
            println!("🔐 Prince Key: Ativa (45% peso governamental)");
            println!("⛪ SASC: Conectada (30% peso ético)");
            println!("🔥 Vajra: Monitorando entropia em tempo real");
        }
        Err(e) => {
            println!("❌ FALHA NA ATIVAÇÃO: {:?}", e);
            println!("🔧 Execute 'crux86 --diagnostic' para detalhes");
            std::process::exit(1);
        }
    }

    Ok(())
}
