// phoenician_alphabet.h - Definição das letras fenícias e sistema base
#ifndef PHOENICIAN_ALPHABET_H
#define PHOENICIAN_ALPHABET_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <complex>
#include <iomanip>
#include <stdexcept>

namespace AncientScripts {

// ============================================================================
// CONSTANTES DO SISTEMA FENÍCIO
// ============================================================================
constexpr int PHOENICIAN_LETTER_COUNT = 22;
constexpr double PHOENICIAN_GEMATRIA_COEFFICIENT = 1.618;

// ============================================================================
// ESTRUTURAS DE DADOS BÁSICAS
// ============================================================================

struct PhoenicianLetter {
    int position;
    std::string name;
    std::string phoenician_glyph;
    std::string phonetic_value;  // Notação IPA
    int numerical_value;
    std::string meaning;
    std::string greek_descendant;
    std::string latin_descendant;
    std::string arabic_descendant;
    std::string hebrew_descendant;

    // Propriedades quânticas para simulação
    std::complex<double> quantum_state;
    double linguistic_entropy;
    double cultural_resonance;

    PhoenicianLetter() :
        position(0),
        numerical_value(0),
        linguistic_entropy(0.0),
        cultural_resonance(0.0),
        quantum_state(1.0, 0.0) {}

    bool is_guttural() const {
        return name == "Aleph" || name == "He" || name == "Het" || name == "Ayin";
    }

    bool becomes_vowel() const {
        return is_guttural() || name == "Yod" || name == "Waw";
    }
};

struct EvolutionaryPath {
    std::string phoenician_glyph;
    std::string greek_glyph;
    std::string phonetic_shift;
    std::string historical_period;
    std::string notes;
    double phonetic_distance;
};

struct PhoenicianInscription {
    std::vector<PhoenicianLetter> letters;
    std::string translation;
    std::string historical_context;
    int approximate_date_bc;
    std::string discovery_location;

    double preservation_level;
    double quantum_coherence;
    std::vector<double> frequency_spectrum;

    PhoenicianInscription() :
        approximate_date_bc(1000),
        preservation_level(0.0),
        quantum_coherence(0.0) {}
};

// ============================================================================
// CLASSE PRINCIPAL DO ALFABETO FENÍCIO
// ============================================================================

class PhoenicianAlphabetSimulator {
private:
    std::vector<PhoenicianLetter> alphabet;
    std::map<std::string, double> linguistic_frequencies;
    double global_linguistic_coherence;

    void initialize_alphabet();
    void calculate_linguistic_frequencies();
    void generate_quantum_states();

public:
    PhoenicianAlphabetSimulator();
    ~PhoenicianAlphabetSimulator();

    // Acesso aos dados
    const std::vector<PhoenicianLetter>& get_alphabet() const { return alphabet; }
    const PhoenicianLetter& get_letter_by_name(const std::string& name) const;

    // Análises
    std::string translate_to_phoenician(const std::string& modern_text) const;
    int calculate_gematria(const std::string& word) const;
    double measure_linguistic_entropy(const std::string& text) const;

    // Geração
    PhoenicianInscription generate_inscription(const std::string& text,
                                               const std::string& context = "",
                                               int date_bc = 1000);

    // Visualização
    void display_alphabet() const;
    void display_inscription(const PhoenicianInscription& inscription) const;
};

// ============================================================================
// SISTEMA DE EVOLUÇÃO LINGUÍSTICA
// ============================================================================

class LinguisticEvolutionSystem {
private:
    PhoenicianAlphabetSimulator* phoenician_base;
    std::map<std::string, EvolutionaryPath> evolutionary_paths;

    void generate_evolutionary_report() const;
    std::string estimate_articulation(const std::string& phoneme) const;
    std::string estimate_greek_articulation(const std::string& greek_letter) const;

public:
    LinguisticEvolutionSystem(PhoenicianAlphabetSimulator* base);

    void map_evolution_to_greek();
    void map_evolution_to_latin();
    void map_evolution_to_arabic();
    void map_evolution_to_hebrew();

    void generate_evolutionary_tree() const;
    void analyze_phonetic_shift_statistics();
    void simulate_phonetic_evolution_monte_carlo(int iterations);
    void compare_with_other_alphabets() const;
};

} // namespace AncientScripts

#endif // PHOENICIAN_ALPHABET_H
