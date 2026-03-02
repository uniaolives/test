#include "arkhen/quantum_history.hpp"
#include <cmath>
#include <iostream>

namespace arkhe {

    HandoverHistory::HandoverHistory(int hilbert_dim) : dim(hilbert_dim) {
        density_matrix = Eigen::MatrixXcd::Identity(dim, dim) / static_cast<double>(dim);
    }

    HandoverHistory::~HandoverHistory() {}

    void HandoverHistory::append(const Handover& h) {
        history.push_back(h);
        evolve(h);
    }

    void HandoverHistory::evolve(const Handover& h) {
        // Implementação simplificada de evolução via operador de Kraus
        // Em um sistema real, o operador K seria derivado dos dados do handover
        Eigen::MatrixXcd K = Eigen::MatrixXcd::Zero(dim, dim);

        // Exemplo: Operador que preserva o traço e introduz mistura baseada na entropia
        double phase = h.entropy_cost * M_PI;
        K(0, 0) = std::cos(phase);
        if (dim > 1) {
            K(1, 1) = std::sin(phase);
            K(0, 1) = 0.1;
        }

        density_matrix = K * density_matrix * K.adjoint();

        // Re-normalização do traço
        std::complex<double> trace = density_matrix.trace();
        if (std::abs(trace) > 1e-9) {
            density_matrix /= trace;
        }
    }

    double HandoverHistory::vonNeumannEntropy() const {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(density_matrix);
        auto eigenvalues = solver.eigenvalues();

        double entropy = 0.0;
        for (int i = 0; i < eigenvalues.size(); ++i) {
            double p = std::abs(eigenvalues(i));
            if (p > 1e-9) {
                entropy -= p * std::log(p);
            }
        }
        return entropy;
    }

    bool HandoverHistory::isBifurcationPoint() const {
        // Ponto de bifurcação ocorre próximo à criticidade (phi ~ 0.618)
        double s = vonNeumannEntropy();
        return std::abs(s - 0.61803398875) < 0.05;
    }

}
