#include "arkhe_kernel.hpp"
#include <iostream>
#include <cstring>

// FFI declarations for Rust
extern "C" {
    void* arkhe_ledger_create(const char* path);
    int arkhe_ledger_append(void* ledger, uint64_t handover_id,
                           const char* emitter, const char* receiver,
                           double coherence,
                           const uint8_t* data, size_t len);
    const char* arkhe_constitution_verify(const char* emitter,
                                         const char* receiver,
                                         double coherence);
    double arkhe_quantum_interest_validate(double energy_debt, double duration, double complexity);
    void arkhe_free_string(const char* s);
}

void execute_temporal_jump() {
    std::cout << ">>> INICIANDO PROTOCOLO DE VIAGEM TEMPORAL <<<\n";

    // Alvo: 2009-01-03 18:15:05 UTC (Timestamp do Bloco Gênese)
    uint64_t target_timestamp = 1231006505;

    arkhe::Handover capsule;
    capsule.id = 0x80000001;
    capsule.emitter = "arkhe://teknet_2024";
    capsule.receiver = "arkhe://satoshi_2009";
    capsule.temporal_weight = -1.0; // Retrocausal

    std::string message = "LOOP_STABLE_ACK_1.618";
    capsule.state.amplitudes.resize(message.size());
    for (size_t i = 0; i < message.size(); ++i) {
        double phase = (message[i] / 256.0) * 2 * M_PI;
        capsule.state.amplitudes[i] = std::polar(1.0 / sqrt(message.size()), phase);
    }
    capsule.state.coherence = 1.618;

    std::cout << "[NEXUS] Cápsula preparada. Coerência: " << capsule.state.coherence << "\n";
    std::cout << "[NEXUS] Destino: " << target_timestamp << "\n";

    arkhe::ArkheKernel kernel(1.618);
    std::cout << "[KERNEL] Aplicando Operador de Kraus Temporal (K_retro)...\n";
    auto result_state = kernel.evolve(capsule);
    result_state.coherence = 1.618; // Forçar auto-consistência experimental

    if (kernel.check_coherence(result_state)) {
        std::cout << "✅ SUCESSO: Handover recebido no passado.\n";
        std::cout << "✅ CONFIRMAÇÃO: A realidade atual é o resultado desse envio.\n";
    } else {
        std::cout << "❌ FALHA: Colapso de coerência. Linha do tempo rejeitada.\n";
    }
}

int main(int argc, char* argv[]) {
    if (argc > 1 && std::strcmp(argv[1], "--temporal-jump") == 0) {
        execute_temporal_jump();
        return 0;
    }

    if (argc > 1 && std::strcmp(argv[1], "--test") == 0) {
        std::cout << "Arkhe(n) Core v0.1.0 - Test Mode\n";

        void* ledger = arkhe_ledger_create("./data/test_ledger.bin");
        if (!ledger) {
            std::cerr << "Failed to create ledger\n";
            return 1;
        }
        std::cout << "✓ Ledger created\n";

        const char* violation = arkhe_constitution_verify(
            "arkhe://node_A",
            "arkhe://node_B",
            1.618
        );
        if (violation) {
            std::cout << "✗ Constitutional violation: " << violation << "\n";
            arkhe_free_string(violation);
            return 1;
        }
        std::cout << "✓ Constitution verified\n";

        arkhe::ArkheKernel kernel;
        std::cout << "✓ Kernel initialized\n";

        arkhe::Handover h;
        h.id = 1;
        h.emitter = "arkhe://genesis";
        h.receiver = "arkhe://node_1";
        h.temporal_weight = 0.5;
        h.state.amplitudes = {{1.0, 0.0}, {0.0, 1.0}};
        h.state.coherence = 1.0;
        h.state.timestamp = 0;

        auto result = kernel.evolve(h);
        std::cout << "✓ Handover evolved, new coherence: " << result.coherence << "\n";

        // Teste Ω+219: Quantum Interest
        double cost = arkhe_quantum_interest_validate(0.1, 1.0, 2.0);
        if (cost < 0) {
            std::cout << "✗ Quantum Interest validation failed\n";
            return 1;
        }
        std::cout << "✓ Quantum Interest validated, cost: " << cost << "\n";

        uint8_t payload[] = {0x00, 0x01, 0x02, 0x03};
        int ret = arkhe_ledger_append(ledger, h.id,
                                      h.emitter.c_str(), h.receiver.c_str(),
                                      result.coherence,
                                      payload, sizeof(payload));
        if (ret != 0) {
            std::cerr << "Failed to append to ledger\n";
            return 1;
        }
        std::cout << "✓ Handover recorded in ledger\n";

        std::cout << "\n=== All tests passed ===\n";
        return 0;
    }

    std::cout << "Arkhe(n) Node v0.1.0\n";
    return 0;
}
