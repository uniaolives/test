#!/usr/bin/env python3
import logging
import signal
import sys
import time
from tim_vm_validator.core import GeometricValidator
from tim_vm_validator.netlink import RegistrationNL

LOG = logging.getLogger("tim-vald")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


class TimValidatorDaemon:
    def __init__(self, proto=31, z_configs=None):
        self.running = True
        self.validator = GeometricValidator(min_acc=0.85, entropy_threshold=4.0)
        self.nl = RegistrationNL(proto=proto, validator=self.validator)
        self.z_configs = z_configs or [(13,), (257,)]

    def handle_signal(self, signum, frame):
        LOG.warning("Signal %d received, shutting down...", signum)
        self.running = False

    def run(self):
        LOG.info("TIM-VALD daemon starting...")
        LOG.info("GeometricValidator initialized (Z_%s)",
                 " + Z_".join(str(z[0]) for z in self.z_configs))
        self.nl.open()

        # simple single-threaded event loop; your stress harness can be threaded
        while self.running:
            self.nl.poll_once(timeout=0.01)
        LOG.info("TIM-VALD daemon stopping...")
        self.nl.close()


def main():
    setup_logging()
    daemon = TimValidatorDaemon()

    signal.signal(signal.SIGINT, daemon.handle_signal)
    signal.signal(signal.SIGTERM, daemon.handle_signal)

    try:
        daemon.run()
    except Exception:
        LOG.exception("Fatal error in daemon")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
