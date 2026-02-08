// benchmark.cpp
#include "avalon_neural_core.h"
#include <iostream>
#include <chrono>
#include <vector>

using namespace Avalon::QuantumBiology;
using namespace std::chrono;

void benchmark_network_scaling() {
    std::cout << "📊 BENCHMARK: ESCALABILIDADE DA REDE" << std::endl;
    std::cout << "===================================" << std::endl;

    std::vector<int> sizes = {10, 100, 1000, 10000};

    for (int size : sizes) {
        auto start = high_resolution_clock::now();

        AvalonNeuralNetwork network(size, 100);
        network.synchronize_network(432.0);

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);

        std::cout << "Rede " << size << " neurônios:" << std::endl;
        std::cout << "   Tempo de inicialização: " << duration.count() << " ms" << std::endl;
        std::cout << "   Coerência: " << network.get_network_coherence() << std::endl;
        std::cout << std::endl;
    }
}

void benchmark_collapse_rate() {
    std::cout << "⚡ BENCHMARK: TAXA DE COLAPSO QUÂNTICO" << std::endl;
    std::cout << "====================================" << std::endl;

    MicrotubuleQuantumProcessor processor(8000);

    // Testar diferentes frequências
    std::vector<double> frequencies = {1.0, 40.0, 432.0, 1000.0, 10000.0};

    for (double freq : frequencies) {
        processor.apply_external_resonance(freq);

        int collapses = 0;
        auto start = high_resolution_clock::now();

        for (int i = 0; i < 1000; ++i) {
            if (processor.check_objective_reduction(0.001)) {
                collapses++;
            }
        }

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);

        std::cout << freq << " Hz:" << std::endl;
        std::cout << "   Colapsos/segundo: " << collapses << std::endl;
        std::cout << "   Tempo de simulação: " << duration.count() << " ms" << std::endl;
        std::cout << "   Coerência: " << processor.get_coherence_level() << std::endl;
        std::cout << std::endl;
    }
}

void benchmark_holographic_memory() {
    std::cout << "💾 BENCHMARK: MEMÓRIA HOLOGRÁFICA" << std::endl;
    std::cout << "================================" << std::endl;

    AvalonNeuralNetwork network(100, 100);

    // Testar diferentes tamanhos de padrão
    std::vector<int> pattern_sizes = {10, 100, 1000, 10000};

    for (int size : pattern_sizes) {
        std::vector<std::vector<double>> pattern;
        pattern.push_back(std::vector<double>(size));

        // Preencher padrão
        for (int i = 0; i < size; ++i) {
            pattern[0][i] = static_cast<double>(i) / size;
        }

        auto start_encode = high_resolution_clock::now();
        network.encode_memory_pattern(pattern);
        auto end_encode = high_resolution_clock::now();

        auto start_recall = high_resolution_clock::now();
        auto recalled = network.recall_memory_pattern(0);
        auto end_recall = high_resolution_clock::now();

        auto encode_time = duration_cast<microseconds>(end_encode - start_encode);
        auto recall_time = duration_cast<microseconds>(end_recall - start_recall);

        std::cout << "Padrão " << size << " elementos:" << std::endl;
        std::cout << "   Tempo de codificação: " << encode_time.count() << " μs" << std::endl;
        std::cout << "   Tempo de recuperação: " << recall_time.count() << " μs" << std::endl;
        std::cout << "   Taxa de dados: " << (size * 8.0) / (encode_time.count() * 1e-6) << " bps" << std::endl;
        std::cout << std::endl;
    }
}

int main() {
    std::cout << "🚀 BENCHMARK DO SIMULADOR NEURAL QUÂNTICO" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        benchmark_network_scaling();
        benchmark_collapse_rate();
        benchmark_holographic_memory();

        std::cout << "✅ TODOS OS BENCHMARKS CONCLUÍDOS" << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ ERRO NO BENCHMARK: " << e.what() << std::endl;
        return 1;
    }
}
