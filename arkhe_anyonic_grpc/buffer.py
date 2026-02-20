# arkhe_anyonic_grpc/buffer.py
# Buffer de emissão para ordenação temporal antes do envio

import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Optional, Callable
import heapq
import grpc

@dataclass(order=True)
class BufferedPacket:
    """Pacote aguardando emissão ordenada"""
    timestamp_ns: int
    alpha_num: int = field(compare=False)
    alpha_den: int = field(compare=False)
    phase: complex = field(compare=False)
    payload: bytes = field(compare=False)
    callback: Callable = field(compare=False)

    def __post_init__(self):
        # Usar timestamp como chave de ordenação
        self._sort_key = self.timestamp_ns

class EmissionBuffer:
    """Buffer que ordena pacotes antes de envio gRPC"""

    def __init__(self, max_size: int = 100, timeout_ms: float = 50.0):
        self.max_size = max_size
        self.timeout_ms = timeout_ms
        self._buffer: List[BufferedPacket] = []
        self._lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None

    async def submit(self,
                     alpha_num: int,
                     alpha_den: int,
                     phase: complex,
                     payload: bytes,
                     callback: Callable) -> None:
        """Submete pacote para emissão ordenada"""
        packet = BufferedPacket(
            timestamp_ns=time.time_ns(),
            alpha_num=alpha_num,
            alpha_den=alpha_den,
            phase=phase,
            payload=payload,
            callback=callback
        )

        async with self._lock:
            heapq.heappush(self._buffer, packet)

            # Iniciar task de flush se não existir
            if self._flush_task is None or self._flush_task.done():
                self._flush_task = asyncio.create_task(self._flush_loop())

    async def _flush_loop(self):
        """Loop que periodicamente flusha pacotes ordenados"""
        while True:
            await asyncio.sleep(self.timeout_ms / 1000.0)

            async with self._lock:
                if not self._buffer:
                    break

                # Coletar pacotes prontos (mais antigos ou buffer cheio)
                ready = []
                while (self._buffer and
                       (len(ready) < self.max_size or
                        self._is_ready(self._buffer[0]))):
                    ready.append(heapq.heappop(self._buffer))

                # Emitir pacotes ordenados
                for pkt in ready:
                    await self._emit(pkt)

    def _is_ready(self, packet: BufferedPacket) -> bool:
        """Verifica se pacote passou do timeout"""
        elapsed_ms = (time.time_ns() - packet.timestamp_ns) / 1_000_000
        return elapsed_ms > self.timeout_ms

    async def _emit(self, packet: BufferedPacket):
        """Emite pacote via gRPC"""
        # Aqui chamaria o interceptor e o stub gRPC real
        await packet.callback(packet)

    async def drain(self):
        """Força emissão de todos os pacotes pendentes"""
        async with self._lock:
            while self._buffer:
                pkt = heapq.heappop(self._buffer)
                await self._emit(pkt)

class AnyonicClientInterceptor:
    """Base class for anyonic client interceptor"""
    def __init__(self, alpha_num: int, alpha_den: int, acc_phase: complex):
        self.alpha_num = alpha_num
        self.alpha_den = alpha_den
        self.acc_phase = acc_phase

# Uso integrado com interceptor
class BufferedAnyonicClientInterceptor(AnyonicClientInterceptor):
    """Interceptor com buffer de emissão ordenada"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._emission_buffer = EmissionBuffer(max_size=50, timeout_ms=20.0)

    async def intercept_unary_unary(self,
                                    continuation: Callable,
                                    client_call_details,
                                    request):
        # Submeter ao buffer em vez de enviar imediatamente
        future = asyncio.Future()

        async def emit_callback(packet):
            # Construir metadados e chamar continuation
            metadata = list(client_call_details.metadata or [])
            metadata.extend([
                ('x-arkhe-timestamp-ns', str(packet.timestamp_ns)),
                ('x-arkhe-alpha-num', str(packet.alpha_num)),
                ('x-arkhe-alpha-den', str(packet.alpha_den)),
                ('x-arkhe-phase-re', str(packet.phase.real)),
                ('x-arkhe-phase-im', str(packet.phase.imag)),
            ])

            new_details = grpc.ClientCallDetails(
                client_call_details.method,
                client_call_details.timeout,
                metadata,
                client_call_details.credentials,
                client_call_details.wait_for_ready,
                client_call_details.compression
            )

            result = continuation(new_details, request)
            future.set_result(result)

        await self._emission_buffer.submit(
            self.alpha_num, self.alpha_den, self.acc_phase,
            b'',  # payload extraído do request
            emit_callback
        )

        return await future
