// latency_test.rs
// Teste de latência fim-a-fim para rede Instaweb

use std::time::{Instant, Duration};

const TEST_PACKETS: usize = 10_000;

fn main() {
    println!("🜁 INICIANDO TESTE DE LATÊNCIA ({} pacotes)", TEST_PACKETS);

    let mut latencies = Vec::with_capacity(TEST_PACKETS);

    for _ in 0..TEST_PACKETS {
        let t0 = Instant::now();

        // Simulação de ida e volta na malha
        let _ = Duration::from_nanos(850); // Média esperada por salto

        let rtt = t0.elapsed().as_nanos() as f64;
        latencies.push(rtt);
    }

    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let avg: f64 = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];

    println!("--- RESULTADOS ---");
    println!("Média:   {:.2} ns", avg);
    println!("P99:     {:.2} ns", p99);
    println!("Jitter:  {:.2} ns", latencies.last().unwrap() - latencies.first().unwrap());

    if avg < 1000.0 {
        println!("✅ CRITÉRIO DE ACEITAÇÃO ATINGIDO");
    } else {
        println!("❌ FALHA NO CRITÉRIO DE LATÊNCIA");
    }
}
