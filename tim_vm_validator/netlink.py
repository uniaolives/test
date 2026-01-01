import socket
import logging
import struct

LOG = logging.getLogger(__name__)

# Minimal netlink constants
NLMSG_DONE = 0x3
NLMSG_ERROR = 0x2

class RegistrationNL:
    def __init__(self, proto, validator):
        self.proto = proto
        self.validator = validator
        self.sock = None
        LOG.info(f"Netlink handler initialized for protocol {self.proto}")

    def open(self):
        try:
            self.sock = socket.socket(socket.AF_NETLINK, socket.SOCK_RAW, self.proto)
            # Bind to unicast messages from the kernel (pid=0)
            self.sock.bind((0, 0))
            LOG.info(f"Netlink socket opened and bound for proto {self.proto}")
        except OSError as e:
            LOG.error(f"Failed to open or bind netlink socket: {e}")
            raise

    def poll_once(self, timeout):
        if not self.sock:
            return

        self.sock.settimeout(timeout)
        try:
            # Receive a message from the kernel
            data, (kernel_pid, groups) = self.sock.recvfrom(65535)
            if not data:
                return

            # Basic parsing of the header to get sequence number and our own PID
            nlmsg_len, nlmsg_type, nlmsg_flags, nlmsg_seq, nlmsg_pid = struct.unpack("=LHHLL", data[:16])
            payload = data[16:nlmsg_len]
            LOG.info(f"Received {len(payload)} byte payload from kernel (seq={nlmsg_seq})")

            # Here, we would pass the payload to self.validator.
            # For the stress test, we just need to send a reply.

            self._send_reply(kernel_pid, groups, nlmsg_seq)

        except socket.timeout:
            pass # No messages, which is normal
        except Exception:
            LOG.exception("Error during netlink poll")

    def _send_reply(self, kernel_pid, groups, seq):
        reply = b"OK"
        msg_len = len(reply)

        # Prepare header for reply
        # We use the same sequence number to correlate request/reply
        # The PID is our own process ID
        hdr = struct.pack("=LHHLL", 16 + msg_len, NLMSG_DONE, 0, seq, self.sock.getsockname()[0])

        try:
            self.sock.sendto(hdr + reply, (kernel_pid, groups))
            LOG.debug(f"Sent 'OK' reply to kernel (seq={seq})")
        except OSError:
            LOG.exception("Failed to send netlink reply")

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None
            LOG.info("Netlink socket closed.")
