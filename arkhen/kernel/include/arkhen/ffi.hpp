#pragma once
#include <cstdint>

extern "C" {

    struct C_Handover {
        uint64_t id;
        uint64_t source_id;
        uint64_t target_id;
        double entropy_cost;
        double half_life;
    };

    typedef void* HandoverHistoryPtr;

    HandoverHistoryPtr arkhe_history_new(int hilbert_dim);
    void arkhe_history_free(HandoverHistoryPtr ptr);

    void arkhe_history_append(HandoverHistoryPtr ptr, C_Handover h);
    double arkhe_history_von_neumann_entropy(HandoverHistoryPtr ptr);
    bool arkhe_history_is_bifurcation(HandoverHistoryPtr ptr);
}
