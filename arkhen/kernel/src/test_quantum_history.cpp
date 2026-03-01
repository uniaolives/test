#include "arkhen/quantum_history.hpp"
#include <iostream>
#include <iomanip>
#include <vector>

using namespace arkhe;

int main() {
    HandoverHistory history(2); // Hilbert space 2D (qubit)

    // Simulate a sequence of handovers
    std::vector<Handover> sequence = {
        {1, 1, 2, 0.3, 1000.0},  // excitatory
        {2, 2, 3, 0.1, 2000.0},  // inhibitory
        {3, 3, 4, 0.5, 3000.0},  // meta
    };

    for (const auto& h : sequence) {
        history.append(h);
        std::cout << "After handover " << h.id
                  << " -> S = " << std::fixed << std::setprecision(4)
                  << history.vonNeumannEntropy() << "\n";
    }

    std::cout << "Is bifurcation point? "
              << (history.isBifurcationPoint() ? "yes" : "no") << "\n";

    return 0;
}
