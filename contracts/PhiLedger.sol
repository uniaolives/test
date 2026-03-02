// SPDX-License-Identifier: MIT
pragma solidity ^0.8.25;

/**
 * @title ΦLedger
 * @dev Registro imutável de decisões Φ seladas
 * Cada transação = um evento de decisão do AGI
 */
contract PhiLedger {
    struct PhiDecision {
        bytes32 delta2_hash;      // Hash do Φ-packet
        uint256 tsc_window;       // Ciclos de clock gastos
        uint8 phi_state;          // 0b01, 0b10, 0b11
        bytes32 attestation_hash; // Hash da prova de selagem
        uint256 block_number;     // Bloco Ethereum para imutabilidade
    }

    mapping(bytes32 => PhiDecision) public phi_history;
    uint256 public decision_count;

    event PhiDecisionRecorded(bytes32 indexed delta2_hash, uint8 phi_state);

    /**
     * @dev Registra decisão Φ apenas se atestada corretamente
     * @param decision Estrutura PhiDecision completa
     * @param signature Assinatura Ed25519 do enclave Rust (Mocked verification)
     */
    function recordPhiDecision(
        PhiDecision calldata decision,
        bytes calldata signature
    ) external {
        // Recalcular hash que foi selado no enclave
        bytes32 recalculated_hash = keccak256(
            abi.encodePacked(
                decision.delta2_hash,
                decision.tsc_window,
                decision.phi_state
            )
        );

        // Em um sistema real, verificaríamos a assinatura Ed25519 aqui.
        // Para a POC, assumimos validade se a assinatura não for vazia.
        require(signature.length > 0, "Attestation invalida");

        // Armazenar (imulavel)
        phi_history[decision.delta2_hash] = decision;
        decision_count++;

        // Evento para indexação
        emit PhiDecisionRecorded(decision.delta2_hash, decision.phi_state);
    }

    /**
     * @dev Função view para auditoria externa
     * Verifica se um Φ-packet foi processado corretamente
     */
    function auditPhiDecision(bytes32 delta2_hash)
        external view returns (bool is_coherent) {
        PhiDecision memory decision = phi_history[delta2_hash];
        return decision.phi_state == 0b01; // Coherent
    }
}
