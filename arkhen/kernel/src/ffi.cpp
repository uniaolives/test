#include "arkhen/ffi.hpp"
#include "arkhen/quantum_history.hpp"

extern "C" {

    HandoverHistoryPtr arkhe_history_new(int hilbert_dim) {
        return new arkhe::HandoverHistory(hilbert_dim);
    }

    void arkhe_history_free(HandoverHistoryPtr ptr) {
        if (ptr != nullptr) {
            delete static_cast<arkhe::HandoverHistory*>(ptr);
        }
    }

    void arkhe_history_append(HandoverHistoryPtr ptr, C_Handover h) {
        auto* history = static_cast<arkhe::HandoverHistory*>(ptr);
        arkhe::Handover cpp_h{h.id, h.source_id, h.target_id, h.entropy_cost, h.half_life};
        history->append(cpp_h);
    }

    double arkhe_history_von_neumann_entropy(HandoverHistoryPtr ptr) {
        auto* history = static_cast<arkhe::HandoverHistory*>(ptr);
        return history->vonNeumannEntropy();
    }

    bool arkhe_history_is_bifurcation(HandoverHistoryPtr ptr) {
        auto* history = static_cast<arkhe::HandoverHistory*>(ptr);
        return history->isBifurcationPoint();
    }
}
