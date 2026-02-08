// test_phoenician.cpp
#include "phoenician_alphabet.h"
#include <cassert>
#include <iostream>

using namespace AncientScripts;

void test_alphabet_initialization() {
    PhoenicianAlphabetSimulator sim;
    auto alphabet = sim.get_alphabet();
    assert(alphabet.size() == 22);
    assert(alphabet[0].name == "Aleph");
    assert(alphabet[21].name == "Taw");
    std::cout << "test_alphabet_initialization PASSED" << std::endl;
}

void test_translation() {
    PhoenicianAlphabetSimulator sim;
    std::string ph = sim.translate_to_phoenician("abc");
    // a -> 𐤀, b -> 𐤁, c -> 𐤂
    assert(ph == "𐤀𐤁𐤂");
    std::cout << "test_translation PASSED" << std::endl;
}

void test_gematria() {
    PhoenicianAlphabetSimulator sim;
    // a (1) + b (2) = 3. Coeff 1.618. 3 * 1.618 = 4.854 -> 4
    int val = sim.calculate_gematria("ab");
    assert(val == 4);
    std::cout << "test_gematria PASSED" << std::endl;
}

int main() {
    test_alphabet_initialization();
    test_translation();
    test_gematria();
    std::cout << "ALL PHOENICIAN TESTS PASSED" << std::endl;
    return 0;
}
