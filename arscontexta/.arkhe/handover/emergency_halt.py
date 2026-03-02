# .arkhe/handover/emergency_halt.py
import sys

class EmergencyHalt:
    """
    Kill switch de emergência com latência < 25ms.
    """

    def __init__(self, safe_core):
        self.safe_core = safe_core

    def trigger(self, reason: str):
        print(f"[CRITICAL] EMERGENCY HALT TRIGGERED: {reason}")
        # Notificar SafeCore para colapso topológico
        self.safe_core._trip(reason)
        # Garantir saída do processo
        sys.exit(1)
