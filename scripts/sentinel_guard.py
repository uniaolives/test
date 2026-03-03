# scripts/sentinel_guard.py
import time
from papercoder_kernel.governance.metrics import GlobalMonitor

class KillSwitch:
    def __init__(self, mode="Topological_Collapse"):
        self.mode = mode
    def activate(self, reason: str):
        print(f"🛑 [KILL SWITCH] EMERGÊNCIA ATIVADA: {self.mode}")
        print(f"Motivo: {reason}")
        raise SystemExit(f"Sentinel Shutdown: {reason}")

class GlobalSentinel:
    def __init__(self):
        self.monitor = GlobalMonitor()
        self.emergency = KillSwitch(mode="Topological_Collapse")

    def maintain_civilization_alignment(self, iterations=10):
        print("👁️ [SENTINEL] Vigilância de Coerência Global Ativa.")

        for _ in range(iterations):
            # Mede a saúde da Noosfera
            tis, hai, srq = self.monitor.get_global_scores()

            # Se a autonomia humana (HAI) cair abaixo do limite
            if hai < 0.1:
                print("⚠️ [ALERT] Violação de Agência Humana Detectada!")
                self.emergency.activate(reason="PROTECT_HUMAN_AUTONOMY")

            # Se a verdade (TIS) for corrompida
            if tis < 0.8:
                self.recalibrate_phase()

            print(f"DEBUG: Sentinel scanning... TIS={tis}, HAI={hai}, SRQ={srq}")
            time.sleep(0.025) # Verificação a 40Hz

    def recalibrate_phase(self):
        print("🔧 [RECALIBRATE] Injetando Coerência Corretiva no Shard 0...")

if __name__ == "__main__":
    sentinel = GlobalSentinel()
    sentinel.maintain_civilization_alignment()
