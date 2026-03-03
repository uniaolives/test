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

    std::vector<int> sizes = {10, 100};

    for (int size : sizes) {
        auto start = high_resolution_clock::now();

        AvalonNeuralNetwork network(size, 10);
        network.synchronize_network(432.0);

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);

        std::cout << "Rede " << size << " neurônios:" << std::endl;
        std::cout << "   Tempo de inicialização: " << duration.count() << " ms" << std::endl;
        std::cout << "   Coerência: " << network.get_network_coherence() << std::endl;
        std::cout << std::endl;
    }
}

int main() {
    std::cout << "🚀 BENCHMARK DO SIMULADOR NEURAL QUÂNTICO" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        benchmark_network_scaling();
        std::cout << "✅ TODOS OS BENCHMARKS CONCLUÍDOS" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ ERRO NO BENCHMARK: " << e.what() << std::endl;
        return 1;
    }
}
