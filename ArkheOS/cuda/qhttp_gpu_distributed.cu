/*
 * qhttp_gpu_distributed.cu
 * Extensão da libqhttp com suporte a múltiplos nós via NCCL/RDMA
 * Implementa emaranhamento global e colapso coordenado.
 */

#include <cuda_runtime.h>
#include <nccl.h>
#include <cuComplex.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STATE_VECTOR_SIZE 16   // 4 qubits
#define MAX_AGENTS 100000
#define NCCL_CHECK(cmd) do {                                 \
    ncclResult_t r = cmd;                                    \
    if (r != ncclSuccess) {                                  \
        printf("NCCL failure %s:%d '%s'\\n",                 \
               __FILE__, __LINE__, ncclGetErrorString(r));   \
        return -1;                                          \
    } } while(0)

typedef cuComplex COMPLEX_TYPE;

typedef struct {
    COMPLEX_TYPE* d_state;        // device pointer
    int agent_id;
    int remote_node;              // -1 se não emaranhado
    int remote_agent_id;
    int bell_type;
} QuantumAgentStateGPU;

static QuantumAgentStateGPU* d_agents = NULL;
static int total_agents = 0;
static ncclComm_t nccl_comm = NULL;
static int node_rank = 0;
static int world_size = 1;

// ----------------------------------------------------------------------
// KERNELS (DEVICE)
// ----------------------------------------------------------------------

__global__ void prepare_bell_state_kernel(COMPLEX_TYPE* state, int type) {
    const float inv_sqrt2 = 0.70710678f;
    // Fix: replaced thread_size with blockDim.x
    // int idx = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = 0; i < STATE_VECTOR_SIZE; i++) {
        state[i] = make_cuFloatComplex(0.0f, 0.0f);
    }

    if (type == 0) { // |Φ+>
        state[0]  = make_cuFloatComplex(inv_sqrt2, 0.0f);
        state[3]  = make_cuFloatComplex(inv_sqrt2, 0.0f);
    } else if (type == 1) { // |Φ->
        state[0]  = make_cuFloatComplex(inv_sqrt2, 0.0f);
        state[3]  = make_cuFloatComplex(-inv_sqrt2, 0.0f);
    } else if (type == 2) { // |Ψ+>
        state[1]  = make_cuFloatComplex(inv_sqrt2, 0.0f);
        state[2]  = make_cuFloatComplex(inv_sqrt2, 0.0f);
    } else { // |Ψ->
        state[1]  = make_cuFloatComplex(inv_sqrt2, 0.0f);
        state[2]  = make_cuFloatComplex(-inv_sqrt2, 0.0f);
    }
}

__device__ int measure_and_collapse(COMPLEX_TYPE* state, curandState* rand_state) {
    float r = curand_uniform(rand_state);
    float cum = 0.0f;
    for (int i = 0; i < STATE_VECTOR_SIZE; i++) {
        float prob = cuCabsf(state[i]);
        prob = prob * prob;
        cum += prob;
        if (r <= cum) {
            for (int j = 0; j < STATE_VECTOR_SIZE; j++)
                state[j] = make_cuFloatComplex(0.0f, 0.0f);
            state[i] = make_cuFloatComplex(1.0f, 0.0f);
            return i;
        }
    }
    return 0;
}

__global__ void collapse_kernel(COMPLEX_TYPE* state, int agent_id, int* result_out) {
    curandState rand_state;
    curand_init(agent_id * 1234, 0, 0, &rand_state);
    *result_out = measure_and_collapse(state, &rand_state);
}

// ----------------------------------------------------------------------
// API HOST
// ----------------------------------------------------------------------

extern "C" {

int qhttp_dist_init(int num_agents, int rank, int size, ncclUniqueId* nccl_id) {
    node_rank = rank;
    world_size = size;
    total_agents = num_agents;

    cudaMalloc(&d_agents, num_agents * sizeof(QuantumAgentStateGPU));
    QuantumAgentStateGPU* h_agents = (QuantumAgentStateGPU*)malloc(num_agents * sizeof(QuantumAgentStateGPU));

    for (int i = 0; i < num_agents; i++) {
        cudaMalloc(&h_agents[i].d_state, STATE_VECTOR_SIZE * sizeof(COMPLEX_TYPE));
        h_agents[i].agent_id = i;
        h_agents[i].remote_node = -1;
        h_agents[i].remote_agent_id = -1;
        h_agents[i].bell_type = 0;

        COMPLEX_TYPE h_state[STATE_VECTOR_SIZE];
        memset(h_state, 0, sizeof(h_state));
        h_state[0] = make_cuFloatComplex(1.0f, 0.0f);
        cudaMemcpy(h_agents[i].d_state, h_state, sizeof(h_state), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_agents, h_agents, num_agents * sizeof(QuantumAgentStateGPU), cudaMemcpyHostToDevice);
    free(h_agents);

    NCCL_CHECK(ncclCommInitRank(&nccl_comm, size, *nccl_id, rank));
    return 0;
}

int qhttp_entangle_remote(int local_agent_id, int remote_rank, int remote_agent_id, int bell_type) {
    if (local_agent_id >= total_agents) return -1;

    QuantumAgentStateGPU h_agent;
    cudaMemcpy(&h_agent, d_agents + local_agent_id, sizeof(QuantumAgentStateGPU), cudaMemcpyDeviceToHost);

    prepare_bell_state_kernel<<<1,1>>>(h_agent.d_state, bell_type);
    cudaDeviceSynchronize();

    h_agent.remote_node = remote_rank;
    h_agent.remote_agent_id = remote_agent_id;
    h_agent.bell_type = bell_type;
    cudaMemcpy(d_agents + local_agent_id, &h_agent, sizeof(QuantumAgentStateGPU), cudaMemcpyHostToDevice);

    int send_data[3] = {local_agent_id, remote_agent_id, bell_type};
    ncclSend(send_data, 3, ncclInt, remote_rank, nccl_comm, 0);

    int ack = 0;
    ncclRecv(&ack, 1, ncclInt, remote_rank, nccl_comm, 0);

    return 0;
}

int qhttp_collapse_remote(int agent_id, int* measured) {
    if (agent_id >= total_agents) return -1;

    QuantumAgentStateGPU h_agent;
    cudaMemcpy(&h_agent, d_agents + agent_id, sizeof(QuantumAgentStateGPU), cudaMemcpyDeviceToHost);

    int* d_result;
    cudaMalloc(&d_result, sizeof(int));
    collapse_kernel<<<1,1>>>(h_agent.d_state, agent_id, d_result);
    cudaMemcpy(measured, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    if (h_agent.remote_node >= 0) {
        ncclSend(measured, 1, ncclInt, h_agent.remote_node, nccl_comm, 0);
        h_agent.remote_node = -1;
        h_agent.remote_agent_id = -1;
        cudaMemcpy(d_agents + agent_id, &h_agent, sizeof(QuantumAgentStateGPU), cudaMemcpyHostToDevice);
    }

    return 0;
}

int qhttp_dist_finalize() {
    if (nccl_comm) ncclCommDestroy(nccl_comm);
    return 0;
}

}
