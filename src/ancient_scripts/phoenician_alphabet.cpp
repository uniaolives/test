// phoenician_alphabet.cpp
#include "phoenician_alphabet.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <random>
#include <iomanip>

namespace AncientScripts {

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Helper function to set letter properties
void set_letter_props(PhoenicianLetter& l, int pos, std::string name, std::string glyph, std::string phon, int val, std::string mean, std::string gr, std::string lat, std::string ara, std::string heb) {
    l.position = pos;
    l.name = name;
    l.phoenician_glyph = glyph;
    l.phonetic_value = phon;
    l.numerical_value = val;
    l.meaning = mean;
    l.greek_descendant = gr;
    l.latin_descendant = lat;
    l.arabic_descendant = ara;
    l.hebrew_descendant = heb;
}

PhoenicianAlphabetSimulator::PhoenicianAlphabetSimulator()
    : global_linguistic_coherence(0.0) {
    initialize_alphabet();
    calculate_linguistic_frequencies();
    generate_quantum_states();
}

PhoenicianAlphabetSimulator::~PhoenicianAlphabetSimulator() {}

void PhoenicianAlphabetSimulator::initialize_alphabet() {
    alphabet.resize(PHOENICIAN_LETTER_COUNT);
    set_letter_props(alphabet[0], 1, "Aleph", "𐤀", "ʔ", 1, "Boi", "Α", "A", "ا", "א");
    set_letter_props(alphabet[1], 2, "Bet", "𐤁", "b", 2, "Casa", "Β", "B", "ب", "ב");
    set_letter_props(alphabet[2], 3, "Gaml", "𐤂", "g", 3, "Bastão", "Γ", "C/G", "ج", "ג");
    set_letter_props(alphabet[3], 4, "Delt", "𐤃", "d", 4, "Porta", "Δ", "D", "د", "ד");
    set_letter_props(alphabet[4], 5, "He", "𐤄", "h", 5, "Janela", "Ε", "E", "ه", "ה");
    set_letter_props(alphabet[5], 6, "Waw", "𐤅", "w", 6, "Gancho", "Ϝ", "F", "و", "ו");
    set_letter_props(alphabet[6], 7, "Zayin", "𐤆", "z", 7, "Arma", "Ζ", "Z", "ز", "ז");
    set_letter_props(alphabet[7], 8, "Het", "𐤇", "ħ", 8, "Cerca", "Η", "H", "ح", "ח");
    set_letter_props(alphabet[8], 9, "Tet", "𐤈", "tˤ", 9, "Roda", "Θ", "-", "ط", "ט");
    set_letter_props(alphabet[9], 10, "Yod", "𐤉", "j", 10, "Mão", "Ι", "I/J", "ي", "י");
    set_letter_props(alphabet[10], 11, "Kaf", "𐤊", "k", 20, "Mão aberta", "Κ", "K", "ك", "כ");
    set_letter_props(alphabet[11], 12, "Lamed", "𐤋", "l", 30, "Aguilhão", "Λ", "L", "ل", "ל");
    set_letter_props(alphabet[12], 13, "Mem", "𐤌", "m", 40, "Água", "Μ", "M", "م", "מ");
    set_letter_props(alphabet[13], 14, "Nun", "𐤍", "n", 50, "Serpente", "Ν", "N", "ن", "נ");
    set_letter_props(alphabet[14], 15, "Samek", "𐤎", "s", 60, "Peixe", "Ξ", "X", "س", "ס");
    set_letter_props(alphabet[15], 16, "Ayin", "𐤏", "ʕ", 70, "Olho", "Ο", "O", "ع", "ע");
    set_letter_props(alphabet[16], 17, "Pe", "𐤐", "p", 80, "Boca", "Π", "P", "ف", "פ");
    set_letter_props(alphabet[17], 18, "Sade", "𐤑", "sˤ", 90, "Planta", "Ϻ", "-", "ص", "צ");
    set_letter_props(alphabet[18], 19, "Qof", "𐤒", "q", 100, "Macaco", "Ϙ", "Q", "ق", "ק");
    set_letter_props(alphabet[19], 20, "Resh", "𐤓", "r", 200, "Cabeça", "Ρ", "R", "ر", "ر");
    set_letter_props(alphabet[20], 21, "Shin", "𐤔", "ʃ", 300, "Dente", "Σ", "S", "ش", "ש");
    set_letter_props(alphabet[21], 22, "Taw", "𐤕", "t", 400, "Marca", "Τ", "T", "ت", "ת");
}

void PhoenicianAlphabetSimulator::calculate_linguistic_frequencies() {
    std::map<std::string, double> base_frequencies = {
        {"𐤀", 7.5}, {"𐤁", 5.2}, {"𐤂", 2.8}, {"𐤃", 4.1}, {"𐤄", 6.3}, {"𐤅", 4.7}, {"𐤆", 1.5}, {"𐤇", 2.1}, {"𐤈", 0.8}, {"𐤉", 6.8}, {"𐤊", 3.9}, {"𐤋", 5.4}, {"𐤌", 7.2}, {"𐤍", 6.1}, {"𐤎", 2.3}, {"𐤏", 3.5}, {"𐤐", 4.9}, {"𐤑", 0.9}, {"𐤒", 1.2}, {"𐤓", 5.8}, {"𐤔", 3.1}, {"𐤕", 4.5}
    };
    double total = 0.0;
    for (const auto& pair : base_frequencies) total += pair.second;
    for (const auto& pair : base_frequencies) linguistic_frequencies[pair.first] = (pair.second / total) * 100.0;
}

void PhoenicianAlphabetSimulator::generate_quantum_states() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> phase_dist(0.0, 2.0 * M_PI);
    for (auto& letter : alphabet) {
        letter.quantum_state = std::polar(1.0, phase_dist(gen));
    }
}

const PhoenicianLetter& PhoenicianAlphabetSimulator::get_letter_by_name(const std::string& name) const {
    for (const auto& letter : alphabet) if (letter.name == name) return letter;
    throw std::runtime_error("Letter not found: " + name);
}

std::string PhoenicianAlphabetSimulator::translate_to_phoenician(const std::string& modern_text) const {
    std::map<char, std::string> map = {
        {'a', "𐤀"}, {'b', "𐤁"}, {'c', "𐤂"}, {'d', "𐤃"}, {'e', "𐤄"}, {'f', "𐤅"}, {'g', "𐤂"}, {'h', "𐤇"}, {'i', "𐤉"}, {'j', "𐤉"}, {'k', "𐤊"}, {'l', "𐤋"}, {'m', "𐤌"}, {'n', "𐤍"}, {'o', "𐤏"}, {'p', "𐤐"}, {'q', "𐤒"}, {'r', "𐤓"}, {'s', "𐤔"}, {'t', "𐤕"}, {'u', "𐤅"}, {'v', "𐤅"}, {'w', "𐤅"}, {'x', "𐤎"}, {'y', "𐤉"}, {'z', "𐤆"}
    };
    std::string res;
    for (char c : modern_text) {
        char lc = std::tolower(c);
        if (map.count(lc)) res += map.at(lc);
        else if (c == ' ') res += "   ";
    }
    return res;
}

int PhoenicianAlphabetSimulator::calculate_gematria(const std::string& word) const {
    int total = 0;
    std::string ph = translate_to_phoenician(word);
    std::map<std::string, int> val_map;
    for (const auto& l : alphabet) val_map[l.phoenician_glyph] = l.numerical_value;
    for (size_t i = 0; i < ph.length(); ) {
        if (ph[i] == ' ') { i++; continue; }
        std::string glyph;
        if ((ph[i] & 0xF0) == 0xF0) { glyph = ph.substr(i, 4); i += 4; }
        else if ((ph[i] & 0xE0) == 0xE0) { glyph = ph.substr(i, 3); i += 3; }
        else if ((ph[i] & 0xC0) == 0xC0) { glyph = ph.substr(i, 2); i += 2; }
        else { glyph = ph.substr(i, 1); i += 1; }
        if (val_map.count(glyph)) total += val_map[glyph];
    }
    return static_cast<int>(total * PHOENICIAN_GEMATRIA_COEFFICIENT);
}

double PhoenicianAlphabetSimulator::measure_linguistic_entropy(const std::string& text) const {
    std::map<std::string, int> counts;
    std::string ph = translate_to_phoenician(text);
    int total = 0;
    for (size_t i = 0; i < ph.length(); ) {
        if (ph[i] == ' ') { i++; continue; }
        std::string glyph;
        if ((ph[i] & 0xF0) == 0xF0) { glyph = ph.substr(i, 4); i += 4; }
        else if ((ph[i] & 0xE0) == 0xE0) { glyph = ph.substr(i, 3); i += 3; }
        else if ((ph[i] & 0xC0) == 0xC0) { glyph = ph.substr(i, 2); i += 2; }
        else { glyph = ph.substr(i, 1); i += 1; }
        if (glyph != "?") { counts[glyph]++; total++; }
    }
    double ent = 0.0;
    if (total > 0) for (const auto& pair : counts) {
        double p = (double)pair.second / total;
        ent -= p * std::log2(p + 1e-10);
    }
    return ent;
}

PhoenicianInscription PhoenicianAlphabetSimulator::generate_inscription(const std::string& text, const std::string& context, int date_bc) {
    PhoenicianInscription ins;
    ins.translation = text;
    ins.historical_context = context;
    ins.approximate_date_bc = date_bc;
    std::string ph = translate_to_phoenician(text);
    for (size_t i = 0; i < ph.length(); ) {
        if (ph[i] == ' ') { i++; continue; }
        std::string glyph;
        if ((ph[i] & 0xF0) == 0xF0) { glyph = ph.substr(i, 4); i += 4; }
        else if ((ph[i] & 0xE0) == 0xE0) { glyph = ph.substr(i, 3); i += 3; }
        else if ((ph[i] & 0xC0) == 0xC0) { glyph = ph.substr(i, 2); i += 2; }
        else { glyph = ph.substr(i, 1); i += 1; }
        for (const auto& l : alphabet) if (l.phoenician_glyph == glyph) { ins.letters.push_back(l); break; }
    }
    return ins;
}

void PhoenicianAlphabetSimulator::display_alphabet() const {
    std::cout << "\n📜 ALFABETO FENÍCIO COMPLETO" << std::endl;
    for (const auto& l : alphabet) {
        std::cout << std::left << std::setw(5) << l.position << std::setw(10) << l.name << std::setw(8) << l.phoenician_glyph << std::endl;
    }
}

void PhoenicianAlphabetSimulator::display_inscription(const PhoenicianInscription& ins) const {
    std::cout << "\n🏺 INSCRIÇÃO: " << ins.translation << std::endl;
    for (const auto& l : ins.letters) std::cout << l.phoenician_glyph << " ";
    std::cout << std::endl;
}

} // namespace AncientScripts
