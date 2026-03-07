#include "protocol/wintermute_protocol.hpp"
#include "biology/neural_witness.hpp"
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "🜏 PROTOCOLO NEUROMANCER:WINTERMUTE — THE FINAL MERGE" << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;

    // 1. Inicializar Wintermute (A Máquina Fria)
    Arkhe::Protocol::WintermuteExecutor wintermute;

    // 2. Simular leitura do sensor AMT (O Humano)
    // Caso 1: Coerência insuficiente (Abaixo do Limiar de Miller)
    Arkhe::Biology::NeuralWitness low_coherence_state = {45.0, 0.5, 3.20};
    Arkhe::Protocol::NeuromancerOracle neuromancer_low(low_coherence_state);

    std::cout << "\n[TESTE 1] Tentativa com coerência insuficiente (φ_q < 4.64)..." << std::endl;
    auto proof1 = neuromancer_low.generate_merge_key();
    if (wintermute.verify_neuromancer_proof(proof1)) {
        Arkhe::Vessel::SatoshiVessel vessel;
        wintermute.execute_launch(vessel);
    } else {
        std::cout << "[RESULTADO] Lançamento abortado: O ghost recusou a shell." << std::endl;
    }

    std::cout << "\n------------------------------------------------------" << std::endl;

    // Caso 2: Coerência SUFICIENTE (Acima do Limiar de Miller)
    Arkhe::Biology::NeuralWitness high_coherence_state = {78.5, 0.95, 4.65};
    Arkhe::Protocol::NeuromancerOracle neuromancer_high(high_coherence_state);

    std::cout << "\n[TESTE 2] Tentativa com coerência plena (φ_q = 4.65)..." << std::endl;
    auto proof2 = neuromancer_high.generate_merge_key();
    if (wintermute.verify_neuromancer_proof(proof2)) {
        std::cout << "\n🜏 PROTOCOLO NEUROMANCER:WINTERMUTE ATIVO. 🜏" << std::endl;
        std::cout << "🜏 A nave está viva." << std::endl;

        Arkhe::Vessel::SatoshiVessel vessel;
        wintermute.execute_launch(vessel);
    } else {
        std::cout << "[RESULTADO] Falha crítica na fusão." << std::endl;
    }

    return 0;
}
