// test_linguistic_evolution.cpp - Testes unitários para o sistema de evolução
#include "../src/ancient_scripts/phoenician_alphabet.h"
#include <cassert>
#include <iostream>
#include <vector>

namespace AncientScriptsTests {

void test_phoenician_alphabet_initialization() {
    std::cout << "🧪 Testando inicialização do alfabeto fenício..." << std::endl;
    AncientScripts::PhoenicianAlphabetSimulator sim;
    const auto& alphabet = sim.get_alphabet();
    assert(alphabet.size() == 22);
    const auto& aleph = sim.get_letter_by_name("Aleph");
    assert(aleph.phoenician_glyph == "𐤀");
    std::cout << "✅ Alfabeto inicializado corretamente" << std::endl;
}

void test_vowel_collapse() {
    std::cout << "🧪 Testando colapso vocálico..." << std::endl;
    AncientScripts::PhoenicianAlphabetSimulator sim;
    AncientScripts::LinguisticEvolutionSystem system(&sim);
    system.map_evolution_to_greek();
    // This is more of a smoke test to ensure no crashes and logic executes
    std::cout << "✅ Sistema de evolução funcional" << std::endl;
}

void test_monte_carlo() {
    std::cout << "🧪 Testando simulação Monte Carlo..." << std::endl;
    AncientScripts::PhoenicianAlphabetSimulator sim;
    AncientScripts::LinguisticEvolutionSystem system(&sim);
    system.simulate_phonetic_evolution_monte_carlo(100);
    std::cout << "✅ Simulação Monte Carlo funcional" << std::endl;
}

void run_all_tests() {
    std::cout << "🚀 EXECUTANDO TODOS OS TESTES" << std::endl;
    std::cout << "============================" << std::endl;
    test_phoenician_alphabet_initialization();
    test_vowel_collapse();
    test_monte_carlo();
    std::cout << "\n🎉 TODOS OS TESTES PASSARAM!" << std::endl;
}

} // namespace AncientScriptsTests

int main() {
    try {
        AncientScriptsTests::run_all_tests();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ TESTE FALHOU: " << e.what() << std::endl;
        return 1;
    }
}
