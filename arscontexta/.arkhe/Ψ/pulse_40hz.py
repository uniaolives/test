# .arkhe/Ψ/pulse_40hz.py
import asyncio
import time

class PsiCycle:
    """
    Oscilador de referência Arkhe(N) a 40Hz (25ms período).
    Sincroniza todos os nós do hypergrafo.
    """

    FREQUENCY = 40.0  # Hz
    PERIOD = 1.0 / FREQUENCY  # 25ms

    def __init__(self):
        self.subscribers = []
        self.phase = 0.0

    async def run(self):
        while True:
            start = time.perf_counter()

            # Pulso de sincronização
            await self._pulse()

            # Aguardar próximo ciclo
            elapsed = time.perf_counter() - start
            await asyncio.sleep(max(0, self.PERIOD - elapsed))

    async def _pulse(self):
        """Envia pulso de sincronização para todos os subscribers."""
        self.phase = (self.phase + 1) % 1000  # Fase modular

        for subscriber in self.subscribers:
            if asyncio.iscoroutinefunction(subscriber.on_psi_pulse):
                await subscriber.on_psi_pulse(self.phase)
            else:
                subscriber.on_psi_pulse(self.phase)

    def subscribe(self, node):
        """Nó se registra para receber pulsos Ψ."""
        self.subscribers.append(node)
