#!/usr/bin/env python3
import socket
import time
import struct
import logging

LOG = logging.getLogger("stress-test")

TIM_REG_NETLINK_PROTO = 31
NLMSG_DONE = 0x3

def main():
    """
    Sends a high volume of netlink messages to the tim-validator daemon
    to simulate the kernel stress test module.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    LOG.info("Starting userspace stress test...")

    try:
        sock = socket.socket(socket.AF_NETLINK, socket.SOCK_RAW, TIM_REG_NETLINK_PROTO)
        # We bind to our own process ID to receive replies
        sock.bind((0, 0))
        sock.settimeout(0.1)
    except OSError as e:
        LOG.error(f"Failed to create and bind netlink socket: {e}")
        LOG.error("Is the 'tim-validator' service running?")
        return

    pid = sock.getsockname()[0]
    count = 0
    errors = 0
    keep_running = True

    LOG.info("Injecting messages at high frequency...")

    try:
        while keep_running:
            payload = f"stress_pid={pid} count={count}".encode('utf-8')
            msg_len = len(payload)

            # Kernel PID is 0. Group is 1 for multicast.
            # However, since we are sending from userspace, we send a unicast
            # message to the kernel's PID (0), which will then be received by
            # any process bound to that protocol (our daemon).

            # The kernel module sent a multicast to group 1.
            # Let's see if we can do that from userspace. The address is (pid, group).
            # We send to the kernel (pid 0) and group 1.

            # Let's try sending directly to the kernel (pid 0, group 0)
            dest_pid = 0
            dest_group = 0

            # Header format: len, type, flags, seq, pid
            hdr = struct.pack("=LHHLL", 16 + msg_len, NLMSG_DONE, 0, count, pid)

            try:
                sock.sendto(hdr + payload, (dest_pid, dest_group))
                count += 1
            except OSError as e:
                LOG.error(f"Failed to send message: {e}")
                errors += 1
                time.sleep(0.1)

            # Briefly check for replies, but don't wait long
            try:
                sock.recvfrom(4096)
            except socket.timeout:
                pass

            if count % 1000 == 0:
                LOG.info(f"{count} requests sent, {errors} errors.")

            if count > 20000: # Stop after 20k messages
                LOG.info("Reached message limit, stopping.")
                keep_running = False

    except KeyboardInterrupt:
        LOG.info("Stress test interrupted.")
    finally:
        sock.close()
        LOG.info(f"Test finished. Total sent: {count}, Total errors: {errors}")


if __name__ == "__main__":
    main()
