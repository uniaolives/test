// cuda/qnet_dpdk.c
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>
#include <rte_mempool.h>
#include <rte_ring.h>
#include <rte_ether.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#define RX_RING_SIZE 1024
#define TX_RING_SIZE 1024
#define NUM_MBUFS 8191
#define MBUF_CACHE_SIZE 250
#define BURST_SIZE 32

typedef struct {
    uint16_t port_id;
    struct rte_mempool *mbuf_pool;
    bool initialized;
    uint64_t tx_count;
    uint64_t rx_count;
} qnet_context_t;

static qnet_context_t g_ctx = {0};

int qnet_init(const char *pci_addr) {
    if (g_ctx.initialized) {
        fprintf(stderr, "qnet already initialized\n");
        return -1;
    }

    // EAL initialization with provided PCI address
    char *argv[] = {
        "qnet",
        "-a", (char*)pci_addr,
        "-l", "0-1",
        "--proc-type=primary",
        "--file-prefix=qnet"
    };
    int argc = sizeof(argv) / sizeof(argv[0]);

    int ret = rte_eal_init(argc, argv);
    if (ret < 0) {
        fprintf(stderr, "EAL init failed: %d\n", ret);
        return -1;
    }

    uint16_t nb_ports = rte_eth_dev_count_avail();
    if (nb_ports == 0) {
        fprintf(stderr, "No Ethernet ports available\n");
        return -1;
    }

    g_ctx.port_id = 0;

    g_ctx.mbuf_pool = rte_pktmbuf_pool_create(
        "MBUF_POOL",
        NUM_MBUFS,
        MBUF_CACHE_SIZE,
        0,
        RTE_MBUF_DEFAULT_BUF_SIZE,
        rte_socket_id()
    );

    if (g_ctx.mbuf_pool == NULL) {
        fprintf(stderr, "Failed to create mbuf pool\n");
        return -1;
    }

    struct rte_eth_conf port_conf = {0};
    port_conf.rxmode.max_lro_pkt_size = RTE_ETHER_MAX_LEN;

    ret = rte_eth_dev_configure(g_ctx.port_id, 1, 1, &port_conf);
    if (ret < 0) return -1;

    ret = rte_eth_rx_queue_setup(g_ctx.port_id, 0, RX_RING_SIZE, rte_eth_dev_socket_id(g_ctx.port_id), NULL, g_ctx.mbuf_pool);
    if (ret < 0) return -1;

    ret = rte_eth_tx_queue_setup(g_ctx.port_id, 0, TX_RING_SIZE, rte_eth_dev_socket_id(g_ctx.port_id), NULL);
    if (ret < 0) return -1;

    ret = rte_eth_dev_start(g_ctx.port_id);
    if (ret < 0) return -1;

    rte_eth_promiscuous_enable(g_ctx.port_id);

    g_ctx.initialized = true;
    printf("qnet initialized on port %u with PCI %s\n", g_ctx.port_id, pci_addr);
    return 0;
}

int qnet_send(const void *data, size_t len) {
    if (!g_ctx.initialized) return -1;

    if (len > RTE_MBUF_DEFAULT_BUF_SIZE - sizeof(struct rte_ether_hdr)) return -1;

    struct rte_mbuf *mbuf = rte_pktmbuf_alloc(g_ctx.mbuf_pool);
    if (mbuf == NULL) return -1;

    // Add Ethernet Header
    struct rte_ether_hdr *eth_hdr = (struct rte_ether_hdr *)rte_pktmbuf_append(mbuf, sizeof(struct rte_ether_hdr) + len);
    memset(&eth_hdr->dst_addr, 0xFF, RTE_ETHER_ADDR_LEN); // Broadcast
    memset(&eth_hdr->src_addr, 0x00, RTE_ETHER_ADDR_LEN);
    eth_hdr->ether_type = rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV4);

    char *payload = (char *)(eth_hdr + 1);
    rte_memcpy(payload, data, len);

    uint16_t nb_tx = rte_eth_tx_burst(g_ctx.port_id, 0, &mbuf, 1);

    if (nb_tx == 0) {
        rte_pktmbuf_free(mbuf);
        return -1;
    }

    g_ctx.tx_count++;
    return (int)len;
}

int qnet_recv(void *buffer, size_t max_len) {
    if (!g_ctx.initialized) return -1;

    struct rte_mbuf *bufs[1];
    // Pull only 1 packet to avoid discarding others in this simple interface
    uint16_t nb_rx = rte_eth_rx_burst(g_ctx.port_id, 0, bufs, 1);

    if (nb_rx == 0) return 0;

    struct rte_mbuf *m = bufs[0];
    struct rte_ether_hdr *eth_hdr = rte_pktmbuf_mtod(m, struct rte_ether_hdr *);

    size_t payload_len = rte_pktmbuf_pkt_len(m);
    if (payload_len > sizeof(struct rte_ether_hdr)) {
        payload_len -= sizeof(struct rte_ether_hdr);
    } else {
        payload_len = 0;
    }

    if (payload_len > max_len) payload_len = max_len;

    if (payload_len > 0) {
        rte_memcpy(buffer, (char*)eth_hdr + sizeof(struct rte_ether_hdr), payload_len);
    }

    rte_pktmbuf_free(m);
    g_ctx.rx_count++;
    return (int)payload_len;
}

void qnet_close() {
    if (g_ctx.initialized) {
        rte_eth_dev_stop(g_ctx.port_id);
        rte_eth_dev_close(g_ctx.port_id);
        rte_eal_cleanup();
        g_ctx.initialized = false;
    }
}
