#ifndef QNET_DPDK_H
#define QNET_DPDK_H

#include <stdint.h>
#include <stddef.h>

/* Minimal header for Parallax Stub simulation */

typedef struct qnet_mbuf {
    void* data;
    size_t len;
} qnet_mbuf_t;

int qnet_init(int argc, char *argv[]);
int qnet_send_packet(uint16_t port_id, void* data, uint16_t len);
int qnet_recv_packet(uint16_t port_id, void* buffer, uint16_t max_len);

#endif
