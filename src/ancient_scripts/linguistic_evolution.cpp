// linguistic_evolution.cpp - Sistema Completo de Evolução Fenício→Grego
#include "phoenician_alphabet.h"
#include <iostream>
#include <map>
#include <vector>
#include <cmath>
#include <fstream>
#include <tuple>
#include <sstream>

namespace AncientScripts {

LinguisticEvolutionSystem::LinguisticEvolutionSystem(PhoenicianAlphabetSimulator* base)
    : phoenician_base(base) {
    std::cout << "🌍 INICIALIZANDO SISTEMA DE EVOLUÇÃO LINGUÍSTICA" << std::endl;
}

void LinguisticEvolutionSystem::map_evolution_to_greek() {
    std::cout << "🏛️ MAPEANDO COLAPSO FONÉTICO: FENÍCIO → GREGO (C. 800 A.C.)" << std::endl;
    std::cout << "==========================================================" << std::endl;

    if (!phoenician_base) {
        std::cout << "❌ Base fenícia não disponível" << std::endl;
        return;
    }

    std::cout << "\n🎯 A GRANDE INOVAÇÃO: CONVERSÃO DE CONSOANTES GUTURAIS EM VOGAIS" << std::endl;

    auto map_vowel_collapse = [&](const std::string& phoe_name,
                                  const std::string& phoe_glyph,
                                  const std::string& phoe_phoneme,
                                  const std::string& greek_name,
                                  const std::string& greek_glyph,
                                  const std::string& greek_phoneme,
                                  const std::string& phonetic_shift) {

        std::cout << "   [VOGAIS] " << phoe_glyph << " " << phoe_name
                  << " (" << phoe_phoneme << ") → "
                  << greek_glyph << " " << greek_name
                  << " (" << greek_phoneme << ")" << std::endl;
        std::cout << "       ↳ " << phonetic_shift << std::endl;

        evolutionary_paths[phoe_name + "→Greek"] = {
            phoe_glyph, greek_glyph, phonetic_shift, "800-700 A.C."
        };
    };

    map_vowel_collapse("Aleph", "𐤀", "ʔ /ʾ/", "Alpha", "Α", "/a/", "Oclusiva glotal surda → Vogal aberta anterior");
    map_vowel_collapse("He", "𐤄", "h /h/", "Epsilon", "Ε", "/e/", "Fricativa glotal surda → Vogal semiaberta anterior");
    map_vowel_collapse("Het", "𐤇", "ħ /ħ/", "Eta", "Η", "/ɛː/", "Fricativa faríngea surda → Vogal aberta-média anterior longa");
    map_vowel_collapse("Yod", "𐤉", "j /j/", "Iota", "Ι", "/i/", "Aproximante palatal → Vogal fechada anterior");
    map_vowel_collapse("Ayin", "𐤏", "ʕ /ʕ/", "Omicron", "Ο", "/o/", "Fricativa faríngea sonora → Vogal semiaberta posterior");

    // 3. CASOS ESPECIAIS E ADAPTAÇÕES
    std::cout << "\n3. CASOS ESPECIAIS E ADAPTAÇÕES:" << std::endl;
    std::cout << "   -----------------------------" << std::endl;

    // Waw (𐤅) → Digamma (Ϝ) e depois desuso parcial
    std::cout << "   [ESPECIAL] 𐤅 Waw (/w/) → Ϝ Digamma (/w/) → Υ Upsilon (/u/)" << std::endl;
    std::cout << "       ↳ Semeivogal labiovelar → Vogal posterior fechada arredondada" << std::endl;

    evolutionary_paths["Waw→Greek_Bifurcation"] = {
        "𐤅", "Ϝ/Υ", "Bifurcação: Semeivogal → [Digamma, Upsilon]", "800-700 A.C."
    };

    std::cout << "\n✅ MAPEAMENTO CONCLUÍDO" << std::endl;
    generate_evolutionary_report();
}

void LinguisticEvolutionSystem::map_evolution_to_latin() { std::cout << "   Mapeando evolução para o Latim..." << std::endl; }
void LinguisticEvolutionSystem::map_evolution_to_arabic() { std::cout << "   Mapeando evolução para o Árabe..." << std::endl; }
void LinguisticEvolutionSystem::map_evolution_to_hebrew() { std::cout << "   Mapeando evolução para o Hebraico..." << std::endl; }

void LinguisticEvolutionSystem::generate_evolutionary_tree() const {
    std::cout << "\n🌳 ÁRVORE EVOLUTIVA DAS ESCRITAS" << std::endl;
    std::cout << "Fenício (1200 A.C.)" << std::endl;
    std::cout << "├── Grego (800 A.C.)" << std::endl;
    std::cout << "│   ├── Latim (700 A.C.)" << std::endl;
    std::cout << "└── Aramaico (800 A.C.)" << std::endl;
}

void LinguisticEvolutionSystem::generate_evolutionary_report() const {
    std::ofstream report_file("output/phoenician_greek_evolution.csv");
    if (report_file.is_open()) {
        std::ostringstream oss;
        oss << "PhoenicianLetter,PhoenicianGlyph,GreekLetter,GreekGlyph,ChangeType\n";
        oss << "Aleph,𐤀,Alpha,Α,GutturalToVowel\n";
        oss << "He,𐤄,Epsilon,Ε,GutturalToVowel\n";
        oss << "Het,𐤇,Eta,Η,GutturalToVowel\n";
        oss << "Yod,𐤉,Iota,Ι,GutturalToVowel\n";
        oss << "Ayin,𐤏,Omicron,Ο,GutturalToVowel\n";
        oss << "Waw,𐤅,Upsilon,Υ,LabiovelarToVowel\n";
        oss << "Waw,𐤅,Digamma,Ϝ,LabiovelarToConsonant\n";

        report_file << oss.str();
        report_file.close();
        std::cout << "   Relatório salvo em: output/phoenician_greek_evolution.csv" << std::endl;
    }
}


} // namespace AncientScripts
