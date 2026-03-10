// arkhe_core/phase_solver.cpp
// O "FFmpeg" da realidade temporal

#include "arkhe_types.h"

/**
 * @brief Solver de alta performance para a equação de dispersão (White Equation)
 * White Equation: d²ρ/dt² = c²∇²ρ - D²∇⁴ρ
 *
 * @param field Estrutura contendo os campos de fase
 * @param D_coeff Coeficiente de dispersão (D)
 * @param dt Passo de tempo
 */
void solve_wave_equation(PhaseField* field, float D_coeff, float dt) {
    // Parallel computation of Laplacian + Biharmonic
    #pragma omp parallel for
    for (int i = 2; i < field->size - 2; ++i) {
        // ∇²ρ (Laplacian)
        float laplacian = field->prev[i-1] - 2*field->prev[i] + field->prev[i+1];

        // ∇⁴ρ (Biharmonic)
        float biharmonic = field->prev[i-2] - 4*field->prev[i-1] + 6*field->prev[i]
                          - 4*field->prev[i+1] + field->prev[i+2];

        // Discretized White Equation:
        // ρ(t+dt) = 2ρ(t) - ρ(t-dt) + (∇²ρ - D²∇⁴ρ) * dt²
        // Note: Assuming c=1 for simplification in this implementation snippet
        field->curr[i] = 2*field->prev[i] - field->prev2[i]
                       + (laplacian - D_coeff * biharmonic) * dt * dt;
    }
}
