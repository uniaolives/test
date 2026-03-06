from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

class ConstitutionalGuard(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Ignorar rotas de saúde ou métricas
        if request.url.path in ["/", "/health", "/docs", "/openapi.json", "/metrics/synchronicity"]:
            return await call_next(request)

        # Verificar Saldo de Coerência (Constante Elena)
        # Mock da verificação H <= 1
        # Em uma implementação real, isso viria de um serviço de monitoramento
        current_h = 0.5 # Exemplo: Sistema saudável

        if current_h > 1.0:
            raise HTTPException(
                status_code=503,
                detail="Constitutional Violation: System Coherence Debt Exceeded (H > 1). Please wait for entropic resolution."
            )

        response = await call_next(request)
        return response
