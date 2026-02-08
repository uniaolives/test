// evolution_analysis.cpp - Análise estatística avançada da evolução
#include "phoenician_alphabet.h"
#include <algorithm>
#include <numeric>
#include <fstream>
#include <iomanip>
#include <random>
#include <iostream>

namespace AncientScripts {

void LinguisticEvolutionSystem::analyze_phonetic_shift_statistics() {
    std::cout << "\n📊 ANÁLISE ESTATÍSTICA DAS MUDANÇAS FONÉTICAS" << std::endl;
    std::cout << "==========================================" << std::endl;

    if (!phoenician_base) return;

    const auto& alphabet = phoenician_base->get_alphabet();
    int total = alphabet.size();
    int vowels = 0;
    int consonants = 0;

    for (const auto& letter : alphabet) {
        if (letter.becomes_vowel()) vowels++;
        else consonants++;
    }

    std::cout << "Total de letras fenícias: " << total << std::endl;
    std::cout << "Tornaram-se vogais: " << vowels << " (" << (double)vowels/total*100 << "%)" << std::endl;
    std::cout << "Permaneceram consoantes: " << consonants << " (" << (double)consonants/total*100 << "%)" << std::endl;
}

void LinguisticEvolutionSystem::simulate_phonetic_evolution_monte_carlo(int iterations) {
    std::cout << "\n🎲 SIMULADOR MONTE CARLO DA EVOLUÇÃO FONÉTICA" << std::endl;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> prob_dist(0.0, 1.0);

    int successful_vowel_shifts = 0;
    double base_vowel_prob = 0.227; // 5/22

    for (int i = 0; i < iterations; ++i) {
        int shifts = 0;
        for (int j = 0; j < 22; ++j) {
            if (prob_dist(gen) < base_vowel_prob) shifts++;
        }
        if (shifts >= 5) successful_vowel_shifts++;
    }

    std::cout << "Simuladas " << iterations << " evoluções." << std::endl;
    std::cout << "Probabilidade de colapso vocálico (>=5 letras): " << (double)successful_vowel_shifts/iterations*100 << "%" << std::endl;
}

void LinguisticEvolutionSystem::compare_with_other_alphabets() const {
    std::cout << "\n🔍 ANÁLISE COMPARATIVA" << std::endl;
    std::cout << "Fenício: 22 letras (Abjad)" << std::endl;
    std::cout << "Grego: 24 letras (Alfabeto)" << std::endl;
    std::cout << "Latim: 26 letras (Alfabeto)" << std::endl;
}

} // namespace AncientScripts
