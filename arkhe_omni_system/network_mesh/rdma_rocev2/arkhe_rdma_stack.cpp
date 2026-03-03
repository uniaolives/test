// arkhe_qutip/network/rdma_stack.cpp
// "Acelerando handovers globais via acesso direto à memória."

#include <iostream>
#include <vector>
#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>

class ArkheRDMAStack {
private:
    struct ibv_context* ctx;
    struct ibv_pd* pd;
    struct ibv_cq* cq;
    struct ibv_qp* qp;
    struct ibv_mr* mr_state;  // Memory region para estados quânticos

    void* hbm_base_addr;
    size_t hbm_size = 8ULL * 1024 * 1024 * 1024;  // 8GB HBM2

public:
    ArkheRDMAStack(void* base_addr) : hbm_base_addr(base_addr) {}

    bool initialize(const char* dev_name) {
        // 1. Abrir dispositivo RDMA (Mellanox ou FPGA ERNIC)
        struct ibv_device** dev_list = ibv_get_device_list(NULL);
        ctx = ibv_open_device(dev_list[0]);

        // 2. Alocar Protection Domain
        pd = ibv_alloc_pd(ctx);

        // 3. Registrar HBM como MR para RDMA (Zero-copy)
        mr_state = ibv_reg_mr(pd, hbm_base_addr, hbm_size,
            IBV_ACCESS_LOCAL_WRITE |
            IBV_ACCESS_REMOTE_READ |

            IBV_ACCESS_REMOTE_WRITE);

        // 4. Criar Completion Queue
        cq = ibv_create_cq(ctx, 128, NULL, NULL, 0);

        // 5. Criar Queue Pair (Reliable Connected)
        struct ibv_qp_init_attr qp_attr = {};
        qp_attr.send_cq = cq;
        qp_attr.recv_cq = cq;
        qp_attr.cap.max_send_wr = 128;
        qp_attr.cap.max_recv_wr = 128;
        qp_attr.cap.max_send_sge = 1;
        qp_attr.qp_type = IBV_QPT_RC;

        qp = ibv_create_qp(pd, &qp_attr);

        std::cout << "🚀 [RDMA] Stack inicializado em " << dev_name << ". HBM registrada." << std::endl;
        return true;
    }

    // Handover Quântico: RDMA WRITE direto para HBM remota
    bool send_handover(uint64_t remote_addr, uint32_t rkey, void* local_data, size_t len) {
        struct ibv_sge sge;
        sge.addr = (uint64_t)local_data;
        sge.length = len;
        sge.lkey = mr_state->lkey;

        struct ibv_send_wr wr = {};
        wr.opcode = IBV_WR_RDMA_WRITE;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.wr.rdma.remote_addr = remote_addr;
        wr.wr.rdma.rkey = rkey;
        wr.num_sge = 1;
        wr.sg_list = &sge;

        struct ibv_send_wr* bad_wr;
        ibv_post_send(qp, &wr, &bad_wr);

        // Poll completion
        struct ibv_wc wc;
        while (ibv_poll_cq(cq, 1, &wc) < 1);

        return wc.status == IBV_WC_SUCCESS;
    }
};
