# src/papercoder_kernel/merkabah/doublezero.py
import subprocess
import json
import logging

logger = logging.getLogger(__name__)

class DoubleZeroInterface:
    """
    Python interface for the DoubleZero CLI.
    """
    def __init__(self, cli_path="doublezero"):
        self.cli_path = cli_path

    def _run_command(self, args):
        try:
            result = subprocess.run([self.cli_path] + args, capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            logger.error(f"DoubleZero CLI error: {e.stderr}")
            raise RuntimeError(f"DoubleZero CLI failed: {e.stderr}")
        except FileNotFoundError:
            logger.error(f"DoubleZero CLI not found at {self.cli_path}")
            return "CLI_NOT_FOUND"

    def keygen(self):
        """Generates a new DoubleZero identity."""
        return self._run_command(["keygen"])

    def get_address(self):
        """Retrieves the DoubleZero address."""
        return self._run_command(["address"])

    def get_latency(self):
        """Checks latency and DZ device discovery."""
        return self._run_command(["latency"])

    def get_status(self):
        """Checks the connection status."""
        return self._run_command(["status"])

    def connect(self):
        """Connects to DoubleZero."""
        return self._run_command(["connect"])

    def disconnect(self):
        """Disconnects from DoubleZero."""
        return self._run_command(["disconnect"])

class DoubleZeroLayer:
    """
    Integrates DoubleZero as a networking/identity layer for Merkabah.
    """
    def __init__(self, interface: DoubleZeroInterface = None):
        self.interface = interface or DoubleZeroInterface()
        self.identity = None

    def initialize(self):
        """Initializes the DoubleZero identity."""
        try:
            self.identity = self.interface.get_address()
            if self.identity == "CLI_NOT_FOUND":
                logger.warning("DoubleZero CLI not available, using simulated identity.")
                self.identity = "SimulatedDoubleZeroAddress11111111111111111111111"
        except Exception as e:
            logger.error(f"Failed to initialize DoubleZero identity: {e}")
            self.identity = "IdentityUnknown"

    def get_info(self):
        return {
            "identity": self.identity,
            "connected": "up" in self.interface.get_status().lower() if self.identity != "IdentityUnknown" else False
        }
