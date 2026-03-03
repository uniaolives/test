#pragma once
#include <vector>
#include <complex>
#include <Eigen/Dense>

namespace arkhe {

    struct Handover {
        uint64_t id;
        uint64_t source_id;
        uint64_t target_id;
        double entropy_cost;
        double half_life;
    };

    class HandoverHistory {
    public:
        HandoverHistory(int hilbert_dim);
        ~HandoverHistory();

        void append(const Handover& h);
        double vonNeumannEntropy() const;
        bool isBifurcationPoint() const;

    private:
        int dim;
        Eigen::MatrixXcd density_matrix;
        std::vector<Handover> history;

        void evolve(const Handover& h);
    };

}
