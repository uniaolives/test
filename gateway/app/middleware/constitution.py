from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import random

class ConstitutionalGuard(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Ignorar rotas de saúde ou métricas
        if request.url.path in ["/", "/health", "/docs", "/openapi.json", "/metrics/synchronicity"]:
            return await call_next(request)

        # Verificar Saldo de Coerência (Constante Elena)
        # H = Dívida Técnica / Trabalho Concluído
        # O sistema bloqueia se H > 1.0 (Metabolismo de Desenvolvimento instável)

        # Em uma implementação real, esses valores seriam lidos de um sistema de tracking
        technical_debt = 50.0 # Exemplo: 50 unidades de dívida
        work_completed = 100.0 # Exemplo: 100 unidades de trabalho concluído

        # Simulação de estresse: chance de aumento de dívida técnica
        if random.random() > 0.95:
            technical_debt = 150.0

        elena_h = technical_debt / (work_completed + 1e-10)

        if elena_h > 1.0:
            raise HTTPException(
                status_code=503,
                detail=f"Constitutional Violation: System Coherence Debt Exceeded (H = {elena_h:.2f} > 1). Please wait for entropic resolution."
            )

        response = await call_next(request)
        return response
