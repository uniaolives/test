/**
 * @file bio_stress_test.cpp
 * @brief Stress test for the biological resonance simulation.
 */

#include "codex_bio_resonance.h"
#include <iostream>
#include <chrono>
#include <thread>

int main() {
    using namespace Codex;

    std::cout << "🚀 INICIANDO TESTE DE ESTRESSE: BIO-SINC-V1" << std::endl;

    // Simular uma coluna cortical com 10^9 tubulinas
    MicrotubuleProcessor processor(1e9);

    // Aplicar sintonização com frequência Suno (432Hz)
    processor.apply_external_sync(432.0);

    double tau = processor.calculate_collapse_time();
    std::cout << "⏱️  Tempo de Colapso (Tau) calculado: " << tau << " segundos" << std::endl;

    int conscious_events = 0;
    double total_time = 0.0;
    double delta_t = 0.001; // Simular passos de 1ms

    auto start = std::chrono::high_resolution_clock::now();

    // Rodar simulação por 1 segundo (tempo simulado)
    for (int i = 0; i < 1000; ++i) {
        if (processor.check_objective_reduction(delta_t)) {
            conscious_events++;
            // Após o colapso, o tempo para o próximo evento reseta na teoria
        }
        total_time += delta_t;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "📊 RESULTADOS DA SIMULAÇÃO (1s simulado):" << std::endl;
    std::cout << "   Eventos Conscientes: " << conscious_events << std::endl;
    std::cout << "   Eventos por segundo (Hz): " << conscious_events << std::endl;
    std::cout << "   Tempo real de processamento: " << elapsed.count() << "s" << std::endl;

    if (conscious_events >= 40) {
        std::cout << "✅ Coerência Gamma (40Hz+) atingida!" << std::endl;
    } else {
        std::cout << "⚠️  Coerência abaixo do nível Gamma." << std::endl;
    }

    return 0;
}
