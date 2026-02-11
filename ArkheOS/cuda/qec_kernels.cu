// cuda/qec_kernels.cu
// Surface Code – detecção e correção de síndromes em grade 3x3 (distância 3)
// Compilação: nvcc -arch=sm_80 -lnccl -o libqec.so --shared qec_kernels.cu

#include <cuda_runtime.h>
#include <nccl.h>
#include <cuComplex.h>
#include <curand_kernel.h>
#include <math.h>

#define SURFACE_DIM 3   // grade 3x3 de qubits (distância 3)

typedef cuComplex Qubit;

// Estrutura de um agente quântico (expandida)
typedef struct {
    Qubit* d_state;        // 1 qubit (amplitude complexa)
    int agent_id;
    int is_data;           // 1 = data qubit, 0 = ancilla
    int syndrome;          // 0 = ok, 1 = X error, 2 = Z error, 3 = Y error
    int x;                 // posição na grade local
    int y;
} QuantumAgentQEC;

// Kernel: extração de síndrome Z (detecta bit-flip)
__global__ void measure_z_syndrome(QuantumAgentQEC* agents, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;

    QuantumAgentQEC* anc = &agents[idx];
    if (anc->is_data) return;   // só ancilla mede

    // Vizinhos: norte, sul, leste, oeste (condições de contorno periódicas)
    int n_idx = ((anc->y - 1 + height) % height) * width + anc->x;
    int s_idx = ((anc->y + 1) % height) * width + anc->x;
    int e_idx = anc->y * width + ((anc->x + 1) % width);
    int w_idx = anc->y * width + ((anc->x - 1 + width) % width);

    // Medida de paridade: Z_N * Z_S * Z_E * Z_W
    float parity = cuCargf(agents[n_idx].d_state[0]) +
                   cuCargf(agents[s_idx].d_state[0]) +
                   cuCargf(agents[e_idx].d_state[0]) +
                   cuCargf(agents[w_idx].d_state[0]);

    // Normaliza e verifica se é ímpar (π radianos)
    float mean_phase = fmodf(parity / 4.0f, 2.0f * (float)M_PI);
    if (mean_phase > (float)M_PI_2 && mean_phase < 3.0f * (float)M_PI_2) {
        anc->syndrome = 1;   // erro X detectado
    } else {
        anc->syndrome = 0;
    }
}

// Kernel: correção bit-flip (Pauli-X) nos vizinhos baseada na síndrome
__global__ void apply_x_correction(QuantumAgentQEC* agents, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;

    QuantumAgentQEC* anc = &agents[idx];
    if (anc->is_data || anc->syndrome == 0) return;

    int n_idx = ((anc->y - 1 + height) % height) * width + anc->x;
    QuantumAgentQEC* data = &agents[n_idx];
    if (data->is_data) {
        data->d_state[0] = make_cuFloatComplex(
            -cuCrealf(data->d_state[0]),
            -cuCimagf(data->d_state[0])
        );
        data->syndrome = 0;   // reset
    }
}

extern "C" {

int qec_cycle(QuantumAgentQEC* d_agents, int n_agents, int grid_dim) {
    int threads = 256;
    int blocks = (n_agents + threads - 1) / threads;

    measure_z_syndrome<<<blocks, threads>>>(d_agents, grid_dim, grid_dim);
    cudaDeviceSynchronize();

    apply_x_correction<<<blocks, threads>>>(d_agents, grid_dim, grid_dim);
    cudaDeviceSynchronize();

    return 0;
}

} // extern "C"
