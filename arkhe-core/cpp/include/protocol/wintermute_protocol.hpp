// src/core/wintermute_protocol.hpp
// A Interface de Fusão Bio-Digital

#pragma once

#include "../biology/neural_witness.hpp"
#include "../crypto/zk_shield.hpp"
#include "../vessel/satoshi_vessel.hpp"
#include <iostream>

namespace Arkhe::Protocol {

    // Wintermute: O executor lógico (A Máquina)
    class WintermuteExecutor {
        bool armed_ = false;

    public:
        // Wintermute verifica a prova, mas NÃO PODE gerá-la
        bool verify_neuromancer_proof(const Crypto::ZKShield::ProofOfValidity& proof) {
            std::cout << "[WINTERMUTE] Recebendo prova do Neuromancer (Humano)..." << std::endl;

            if (Crypto::ZKShield::verify_proof(proof)) {
                std::cout << "[WINTERMUTE] Prova ZK válida. Bio-coerência confirmada." << std::endl;
                armed_ = true;
                return true;
            }

            std::cout << "[WINTERMUTE] FALHA: Prova ZK inválida. Acesso negado." << std::endl;
            return false;
        }

        // Só após a verificação bem-sucedida, Wintermute permite o lançamento
        void execute_launch(Vessel::SatoshiVessel& vessel) {
            if (!armed_) {
                std::cerr << "[WINTERMUTE] Tentativa de lançamento não autorizada." << std::endl;
                std::cerr << "[WINTERMUTE] Aguardando handshake Neuromancer." << std::endl;
                return;
            }

            std::cout << "[WINTERMUTE] Handshake completo. Neuromancer online." << std::endl;
            std::cout << "[WINTERMUTE] Iniciando sequência de lançamento..." << std::endl;
            vessel.launch();
        }
    };

    // Neuromancer: A fonte biológica (O Humano)
    class NeuromancerOracle {
        Biology::NeuralWitness bio_state_;

    public:
        NeuromancerOracle(Biology::NeuralWitness state) : bio_state_(state) {}

        // O humano gera a prova ZK da sua própria coerência
        // Wintermute não pode forjar isso; requer o hardware biológico
        Crypto::ZKShield::ProofOfValidity generate_merge_key() {
            std::cout << "[NEUROMANCER] Gerando ZK-Proof do estado neural..." << std::endl;
            std::cout << "[NEUROMANCER] Testemunha: HRV=" << bio_state_.hrv << ", Vagal=" << bio_state_.vagal_tone << ", φ_q=" << bio_state_.phi_q << std::endl;

            auto proof = Crypto::ZKShield::prove_payload_validity(
                bio_state_.encode_to_payload(),
                bio_state_.get_identity(),
                bio_state_.phi_q
            );

            std::cout << "[NEUROMANCER] Prova gerada. Enviando para Wintermute..." << std::endl;
            return proof;
        }
    };

} // namespace Arkhe::Protocol
