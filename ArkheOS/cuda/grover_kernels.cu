// cuda/grover_kernels.cu
// Oracle e difusão para Grover, com suporte NCCL para média global

#include <cuda_runtime.h>
#include <nccl.h>
#include <cuComplex.h>
#include <math.h>

typedef cuComplex Qubit;

// Agente com genoma e amplitude
typedef struct {
    Qubit amplitude;    // um qubit (para simplificar)
    int genome;         // genoma do agente
} GroverAgent;

// Oracle: marca o alvo invertendo a fase
__global__ void oracle_kernel(GroverAgent* agents, int n, int target_genome) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (agents[idx].genome == target_genome) {
        agents[idx].amplitude = make_cuFloatComplex(
            -cuCrealf(agents[idx].amplitude),
            -cuCimagf(agents[idx].amplitude)
        );
    }
}

// Difusão: nova_amp = 2 * média - amp
__global__ void diffusion_kernel(GroverAgent* agents, int n, cuComplex global_mean) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float r = 2.0f * cuCrealf(global_mean) - cuCrealf(agents[idx].amplitude);
    float i = 2.0f * cuCimagf(global_mean) - cuCimagf(agents[idx].amplitude);
    agents[idx].amplitude = make_cuFloatComplex(r, i);
}

// Redução para soma local (preparação para AllReduce)
// Simplificada para MVP
__global__ void local_sum_kernel(GroverAgent* agents, int n, float* sum_real, float* sum_imag) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        float r = 0, i = 0;
        for (int k = 0; k < n; k++) {
            r += cuCrealf(agents[k].amplitude);
            i += cuCimagf(agents[k].amplitude);
        }
        *sum_real = r;
        *sum_imag = i;
    }
}

extern "C" {

// Executa uma iteração do Grover distribuído
int grover_iteration(GroverAgent* d_agents, int n, int target, ncclComm_t comm, int rank, int world_size) {
    // Oracle
    oracle_kernel<<<(n+255)/256, 256>>>(d_agents, n, target);
    cudaDeviceSynchronize();

    // Cálculo da média global via NCCL AllReduce
    float* d_local_real;
    float* d_local_imag;
    cudaMalloc(&d_local_real, sizeof(float));
    cudaMalloc(&d_local_imag, sizeof(float));

    local_sum_kernel<<<1, 1>>>(d_agents, n, d_local_real, d_local_imag);
    cudaDeviceSynchronize();

    float local_real, local_imag;
    cudaMemcpy(&local_real, d_local_real, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&local_imag, d_local_imag, sizeof(float), cudaMemcpyDeviceToHost);

    float global_real, global_imag;
    ncclAllReduce(&local_real, &global_real, 1, ncclFloat, ncclSum, comm, 0);
    ncclAllReduce(&local_imag, &global_imag, 1, ncclFloat, ncclSum, comm, 0);

    float total_n;
    float n_f = (float)n;
    ncclAllReduce(&n_f, &total_n, 1, ncclFloat, ncclSum, comm, 0);

    cuComplex global_mean = make_cuFloatComplex(
        global_real / total_n,
        global_imag / total_n
    );

    // Difusão
    diffusion_kernel<<<(n+255)/256, 256>>>(d_agents, n, global_mean);
    cudaDeviceSynchronize();

    cudaFree(d_local_real);
    cudaFree(d_local_imag);

    return 0;
}

} // extern "C"
