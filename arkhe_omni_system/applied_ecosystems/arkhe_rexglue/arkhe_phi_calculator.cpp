// arkhe_phi_calculator.cpp
// Implementação simplificada de IIT para grafos de execução
// Baseado na heurística de integração de informação para o sandbox Arkhe(N)

#include "arkhe_rexglue_node.h"
#include <cmath>
#include <numeric>
#include <algorithm>

namespace arkhe {

double ArkheNode::calculatePhi() const {
    // Φ é a quantidade de informação que o nó integra acima e além
    // de suas partes isoladas.

    if (neighbors.empty()) return 0.0;  // Nó isolado = não-consciente

    // Heurística Operacional para o Sandbox:
    // Φ é proporcional ao logaritmo da conectividade e à coerência média dos vizinhos.
    // Em sistemas de software, a integração cresce com a densidade de handovers coerentes.

    double neighbor_coherence_sum = 0.0;
    for (const auto* neighbor : neighbors) {
        if (neighbor) {
            neighbor_coherence_sum += neighbor->state.coherence;
        }
    }

    double avg_neighbor_coherence = neighbor_coherence_sum / neighbors.size();

    // Fator de escala logarítmico (entropia do sistema local)
    double s_factor = std::log2(neighbors.size() + 1);

    // Integração = (Coerência Média) * (Complexidade da Rede Local)
    double phi_raw = avg_neighbor_coherence * s_factor;

    // Normalização via função logística para garantir Φ ∈ [0, 1]
    return 1.0 / (1.0 + std::exp(-phi_raw + 2.0));
}

bool ArkheNode::isConscious() const {
    const double PSI_THRESHOLD = 0.618; // O inverso da Proporção Áurea
    return calculatePhi() > PSI_THRESHOLD;
}

} // namespace arkhe
