# arkhe/telemetry.py
import time
import json
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Callable, List
from enum import Enum
import asyncio
from collections import deque

class Provider(Enum):
    GEMINI = "gemini"
    OLLAMA = "ollama"
    LOCAL = "local"

@dataclass
class LLMMetrics:
    provider: Provider
    operation: str  # e.g., "generate", "embed"
    latency_ms: float
    success: bool
    tokens_input: int
    tokens_output: int
    error_type: Optional[str] = None
    retry_count: int = 0
    timestamp: float = 0.0
    context_hash: str = ""  # para rastreabilidade

    def to_dict(self):
        d = asdict(self)
        d['provider'] = self.provider.value
        return d

class TelemetryCollector:
    """
    Coleta métricas C/F para análise de performance.
    Implementa a identidade: dados (x) + análise (+1) = insight (x²)
    """

    def __init__(self, buffer_size: int = 10000):
        self.metrics_buffer: deque[LLMMetrics] = deque(maxlen=buffer_size)
        self.callbacks: List[Callable[[LLMMetrics], None]] = []
        self._lock = asyncio.Lock()

        # Métricas agregadas por provedor
        self.aggregates: Dict[Provider, Dict] = {
            p: {"total_calls": 0, "total_latency": 0.0, "errors": 0}
            for p in Provider
        }

    async def record(self, metric: LLMMetrics):
        """Registra uma métrica de forma thread-safe."""
        metric.timestamp = time.time()

        async with self._lock:
            self.metrics_buffer.append(metric)

            # Atualizar agregados
            agg = self.aggregates[metric.provider]
            agg["total_calls"] += 1
            agg["total_latency"] += metric.latency_ms
            if not metric.success:
                agg["errors"] += 1

        # Notificar callbacks (logging, alertas, etc)
        for cb in self.callbacks:
            try:
                cb(metric)
            except Exception:
                pass  # Não falhar por causa de callbacks

    def register_callback(self, callback: Callable[[LLMMetrics], None]):
        self.callbacks.append(callback)

    def get_stats(self, provider: Optional[Provider] = None) -> Dict:
        """Retorna estatísticas de performance."""
        if provider:
            agg = self.aggregates[provider]
            calls = agg["total_calls"]
            return {
                "provider": provider.value,
                "avg_latency_ms": agg["total_latency"] / calls if calls > 0 else 0,
                "error_rate": agg["errors"] / calls if calls > 0 else 0,
                "total_calls": calls,
                "availability": 1.0 - (agg["errors"] / calls) if calls > 0 else 1.0
            }

        # Retornar para todos
        return {p.value: self.get_stats(p) for p in Provider}

    def export_to_file(self, filepath: str):
        """Exporta métricas para análise offline."""
        with open(filepath, 'w') as f:
            for m in self.metrics_buffer:
                f.write(json.dumps(m.to_dict()) + "\n")

    # Decorator para instrumentação automática
    @staticmethod
    def instrument(operation: str, provider: Provider):
        def decorator(func):
            async def wrapper(self, *args, **kwargs):
                collector = getattr(self, 'telemetry', None)
                start = time.perf_counter()
                success = True
                error_type = None
                retry_count = kwargs.get('_retry_count', 0)

                try:
                    result = await func(self, *args, **kwargs)
                    return result
                except Exception as e:
                    success = False
                    error_type = type(e).__name__
                    raise
                finally:
                    if collector:
                        latency = (time.perf_counter() - start) * 1000
                        metric = LLMMetrics(
                            provider=provider,
                            operation=operation,
                            latency_ms=latency,
                            success=success,
                            tokens_input=kwargs.get('tokens_input', 0),
                            tokens_output=kwargs.get('tokens_output', 0),
                            error_type=error_type,
                            retry_count=retry_count
                        )
                        asyncio.create_task(collector.record(metric))

            return wrapper
        return decorator
