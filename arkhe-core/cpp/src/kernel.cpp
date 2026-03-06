#include "arkhe_kernel.hpp"
#include <cmath>
#include <algorithm>

namespace arkhe {

ArkheKernel::ArkheKernel(double lambda_target)
    : lambda_target_(lambda_target) {}

FieldState ArkheKernel::evolve(const Handover& input) {
    FieldState output = input.state;

    apply_temporal_operator(output, input.temporal_weight);

    if (output.coherence > lambda_target_ * 0.9) {
        double xi = (output.coherence - lambda_target_) / (lambda_target_ + 1e-10);
        apply_retrocausal_squeeze(output, xi);
    }

    double norm_sq = 0.0;
    for (const auto& amp : output.amplitudes) {
        norm_sq += std::norm(amp);
    }
    output.coherence = (std::sqrt(norm_sq) / (output.amplitudes.size() + 1e-10)) * 1.618;
    output.timestamp = input.timestamp + 1;

    return output;
}

bool ArkheKernel::check_coherence(const FieldState& state) const {
    return state.coherence >= lambda_target_;
}

void ArkheKernel::apply_temporal_operator(FieldState& state, double dt) {
    for (auto& amp : state.amplitudes) {
        double phase = -dt * std::arg(amp);
        amp *= std::polar(1.0, phase);
    }
}

void ArkheKernel::apply_retrocausal_squeeze(FieldState& state, double xi) {
    if (state.amplitudes.size() < 2) return;

    for (size_t i = 0; i < state.amplitudes.size() - 1; ++i) {
        auto a = state.amplitudes[i];
        auto a_dag = state.amplitudes[i + 1];

        double cosh_xi = std::cosh(xi);
        double sinh_xi = std::sinh(xi);

        state.amplitudes[i] = a * cosh_xi + std::conj(a_dag) * sinh_xi;
    }
}

void ArkheKernel::load_quantum_state(const std::vector<std::complex<double>>& qubits) {
    current_state_.amplitudes = qubits;
    current_state_.coherence = 1.0;
    current_state_.timestamp = 0;
}

} // namespace arkhe
