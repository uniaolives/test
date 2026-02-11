// cuda/dpdk_hello.c
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>
#include <stdio.h>

#define NUM_MBUFS 8191
#define MBUF_CACHE_SIZE 250

int main(int argc, char **argv) {
    // Inicializa EAL
    int ret = rte_eal_init(argc, argv);
    if (ret < 0)
        rte_exit(EXIT_FAILURE, "EAL init failed\n");

    // Cria mempool
    struct rte_mempool *mbuf_pool = rte_pktmbuf_pool_create("MBUF_POOL",
        NUM_MBUFS, MBUF_CACHE_SIZE, 0, RTE_MBUF_DEFAULT_BUF_SIZE,
        rte_socket_id());

    if (mbuf_pool == NULL)
        rte_exit(EXIT_FAILURE, "Mempool creation failed\n");

    printf("DPDK initialized successfully!\n");
    printf("Available ports: %u\n", rte_eth_dev_count_avail());

    rte_eal_cleanup();
    return 0;
}
