# .arkhe/coherence/safe_core.py

class SafeCore:
    """
    Circuito de segurança Arkhe(N).
    Interrompe execução em < 25ms se limiares violados.
    """

    def __init__(self):
        self.phi_threshold = 0.1
        self.coherence_min = 0.847
        self.latency_max_ms = 25

        # Estado do circuito
        self.armed = True
        self.tripped = False

    def check(self, phi: float, coherence: float) -> bool:
        """
        Verificação de segurança. Retorna True se seguro, False se kill switch ativado.
        """
        if not self.armed:
            return False

        if phi > self.phi_threshold:
            self._trip(f"Phi exceeded: {phi} > {self.phi_threshold}")
            return False

        if coherence < self.coherence_min:
            self._trip(f"Coherence collapsed: {coherence} < {self.coherence_min}")
            return False

        return True

    def _trip(self, reason: str):
        """Ativa kill switch."""
        self.tripped = True
        self.armed = False

        # Log imediato no ledger
        self._emergency_log(reason)

        # Notificar todos os nós
        self._broadcast_halt()

        # Parada física (simulada ou real)
        raise SystemExit(f"[SAFE CORE] HALT: {reason}")

    def on_psi_pulse(self, phase):
        """Chamado a cada pulso Ψ."""
        # Em uma implementação real, o SafeCore verificaria o estado global aqui.
        pass

    def _emergency_log(self, reason: str):
        """Log de emergência no ledger."""
        print(f"[EMERGENCY LOG] {reason}")

    def _broadcast_halt(self):
        """Notifica todos os nós para pararem."""
        print("[BROADCAST] HALT ALL NODES")
