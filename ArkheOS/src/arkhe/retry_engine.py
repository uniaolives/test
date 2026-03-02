# arkhe/retry_engine.py
import asyncio
import random
from typing import TypeVar, Callable, Any, Optional, List
from enum import Enum
import logging

T = TypeVar('T')

class RetryStrategy(Enum):
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    FIBONACCI = "fibonacci"  # Homenagem à identidade x²=x+1

class RetryConfig:
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
        jitter: bool = True,
        retryable_errors: Optional[List[type]] = None
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.strategy = strategy
        self.jitter = jitter
        self.retryable_errors = retryable_errors or [
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
            Exception # Padrão para simulação
        ]

class RetryEngine:
    """
    Motor de retry com backoff exponencial.
    Implementa: tentativa (x) + espera (+1) = sucesso (x²)
    """

    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
        self.logger = logging.getLogger("arkhe.retry")

    def _calculate_delay(self, attempt: int) -> float:
        """Calcula delay baseado na estratégia."""
        cfg = self.config

        if cfg.strategy == RetryStrategy.FIXED:
            delay = cfg.base_delay
        elif cfg.strategy == RetryStrategy.EXPONENTIAL:
            delay = cfg.base_delay * (2 ** attempt)
        elif cfg.strategy == RetryStrategy.FIBONACCI:
            # Fibonacci: 1, 1, 2, 3, 5, 8...
            fib = self._fibonacci(attempt + 1)
            delay = cfg.base_delay * fib
        else:
            delay = cfg.base_delay

        # Cap no max_delay
        delay = min(delay, cfg.max_delay)

        # Jitter: ±25% aleatório para evitar thundering herd
        if cfg.jitter:
            delay *= random.uniform(0.75, 1.25)

        return delay

    def _fibonacci(self, n: int) -> int:
        if n <= 0: return 0
        if n == 1: return 1
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b

    async def execute(
        self,
        operation: Callable[..., Any],
        *args,
        **kwargs
    ) -> T:
        """
        Executa operação com retry.
        """
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                self.logger.debug(f"Attempt {attempt + 1}/{self.config.max_retries + 1}")
                result = await operation(*args, **kwargs)
                return result

            except Exception as e:
                last_exception = e

                # Verificar se erro é retryable
                if not any(isinstance(e, err) for err in self.config.retryable_errors):
                    self.logger.error(f"Non-retryable error: {e}")
                    raise

                if attempt < self.config.max_retries:
                    delay = self._calculate_delay(attempt)
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(f"All retries exhausted: {e}")

        raise last_exception

    # Rate limiting adaptativo
    async def execute_with_rate_limit(
        self,
        operation: Callable[..., Any],
        rate_limiter: 'RateLimiter',
        *args,
        **kwargs
    ) -> T:
        """Executa com controle de taxa adaptativo."""
        await rate_limiter.acquire()
        return await self.execute(operation, *args, **kwargs)

class RateLimiter:
    """
    Token bucket para rate limiting.
    """
    def __init__(self, rate: float, capacity: int):
        self.rate = rate  # tokens por segundo
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = asyncio.get_event_loop().time()
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1
