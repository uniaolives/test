// main_phoenician.cpp
#include "phoenician_alphabet.h"
#include <iostream>

using namespace AncientScripts;

void run_phoenician_simulation() {
    std::cout << "🏛️ SIMULADOR DO ALFABETO FENÍCIO v1.0" << std::endl;
    std::cout << "====================================" << std::endl;

    // 1. Inicializar simulador
    PhoenicianAlphabetSimulator phoenician_sim;

    // 2. Mostrar alfabeto
    phoenician_sim.display_alphabet();

    // 3. Sistema de evolução linguística
    LinguisticEvolutionSystem evolution_system(&phoenician_sim);
    evolution_system.map_evolution_to_greek();
    evolution_system.generate_evolutionary_tree();

    // 4. Análises estatísticas
    evolution_system.analyze_phonetic_shift_statistics();
    evolution_system.simulate_phonetic_evolution_monte_carlo(10000);
    evolution_system.compare_with_other_alphabets();

    // 5. Testes de tradução e gematria
    std::vector<std::string> test_words = {"aleph", "bet", "shalom", "mlk"};
    std::cout << "\n🔤 TESTES DE TRADUÇÃO:" << std::endl;
    for (const auto& word : test_words) {
        std::cout << "   " << word << " -> " << phoenician_sim.translate_to_phoenician(word)
                  << " (Gematria: " << phoenician_sim.calculate_gematria(word) << ")" << std::endl;
    }

    // 6. Inscrições
    PhoenicianInscription ins = phoenician_sim.generate_inscription("mlk ethbaal", "Byblos", 850);
    phoenician_sim.display_inscription(ins);

    std::cout << "\n✅ SIMULAÇÃO CONCLUÍDA COM SUCESSO" << std::endl;
}

int main() {
    try {
        run_phoenician_simulation();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ ERRO: " << e.what() << std::endl;
        return 1;
    }
}
