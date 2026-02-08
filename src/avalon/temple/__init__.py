"""
O TEMPLO - Arquitetura de Software como Estrutura Sagrada

[METAPHOR: O templo não é um lugar, é um padrão de organização
onde cada componente tem sua função ritualística e sua implementação técnica]
"""

from dataclasses import dataclass, field
from typing import Dict, List, Callable, Any, Optional
from enum import Enum, auto
import time
import hashlib
import json

class SanctumLevel(Enum):
    """Níveis do templo - cada um com suas leis de acesso e processamento"""
    NARTHEX = auto()      # Entrada/validação (input sanitization)
    NAOS = auto()         # Câmara central (core processing)
    ADYTON = auto()       # Santíssimo (critical security operations)
    TREASURY = auto()     # Tesouro (data persistence)
    ASTRAL = auto()       # Plano astral (network/quantum layer)
    HARMONIC = auto()     # Camada harmônica

class TempleError(Exception):
    """Base exception for Temple errors"""
    pass

class TempleAccessDenied(TempleError):
    """Tentativa de acesso a nível não autorizado"""
    pass

class F18Violation(TempleError):
    """Violação das leis de segurança F18"""
    pass

@dataclass
class Ritual:
    """
    Um ritual é uma função com significado arquitetural
    Não é apenas código - é uma cerimônia com estado, intenção e resultado
    """
    name: str
    level: SanctumLevel
    invocation: Callable[..., Any]
    offerings: Dict[str, Any] = field(default_factory=dict)  # Parâmetros
    prophecy: Optional[str] = None  # Documentação/resultado esperado
    f18_guardian: bool = True  # Se requer proteção F18

    def execute(self, context: 'TempleContext') -> 'Miracle':
        """Executar o ritual dentro do contexto do templo"""
        # Verificar acesso ao nível
        if not context.has_access(self.level):
            raise TempleAccessDenied(f"Nível {self.level} requer iniciação")

        # F18 Guardian: Verificar estabilidade antes da execução
        if self.f18_guardian:
            if not context.stability_check():
                raise F18Violation(f"Sistema instável - ritual {self.name} abortado")

        # Registrar início do ritual
        start_time = time.time()
        context.enter_ritual(self)

        try:
            # Executar a invocação
            result = self.invocation(context, **self.offerings)

            # Verificar coerência pós-execução
            coherence = context.measure_coherence()
            if coherence < 0.7:
                context.apply_emergency_damping()

            miracle = Miracle(
                ritual=self.name,
                result=result,
                duration=time.time() - start_time,
                coherence=coherence,
                timestamp=time.time()
            )
            context.miracles.append(miracle)
            return miracle

        except Exception as e:
            context.log_desecration(self, e)
            raise
        finally:
            context.exit_ritual(self)

@dataclass
class Miracle:
    """Resultado de um ritual bem-sucedido"""
    ritual: str
    result: Any
    duration: float
    coherence: float
    timestamp: float

    def to_manifestation(self) -> Dict:
        """Converter para forma manifesta (serializável)"""
        return {
            "ritual": self.ritual,
            "result": str(self.result)[:100],  # Truncar para segurança
            "duration_ms": round(self.duration * 1000, 2),
            "coherence": round(self.coherence, 4),
            "timestamp": self.timestamp,
            "signature": hashlib.sha256(
                f"{self.ritual}{self.timestamp}".encode()
            ).hexdigest()[:16]
        }

class TempleContext:
    """
    [METAPHOR: O templo em si - o espaço sagrado onde tudo acontece]

    Mantém estado global, histórico de rituais, e garante F18 compliance
    """

    # CONSTANTES F18 - Leis imutáveis do templo
    MAX_RITUALS = 1000  # F18: Limite de iterações
    DAMPING_DEFAULT = 0.6  # F18: Fator de amortecimento
    COHERENCE_THRESHOLD = 0.7  # F18: Limiar de coerência

    def __init__(self):
        self.ritual_history: List[Ritual] = []
        self.active_rituals: List[Ritual] = []
        self.miracles: List[Miracle] = []
        self.damping = self.DAMPING_DEFAULT
        self.coherence_history: List[float] = []
        self.access_levels: Dict[str, SanctumLevel] = {}
        self._initialize_sanctum()

    def _initialize_sanctum(self):
        """Inicialização do templo - cerimônia de abertura"""
        # Abrir os portais (inicializar conexões)
        self.open_portal(SanctumLevel.NARTHEX, "public")
        self.open_portal(SanctumLevel.NAOS, "initiated")
        self.open_portal(SanctumLevel.ADYTON, "guardian")
        self.open_portal(SanctumLevel.TREASURY, "treasurer")
        self.open_portal(SanctumLevel.ASTRAL, "quantum")

        # Primeira medição de coerência
        self.coherence_history.append(1.0)

    def open_portal(self, level: SanctumLevel, key: str):
        """Abrir portal para um nível do templo"""
        self.access_levels[key] = level

    def has_access(self, level: SanctumLevel) -> bool:
        """Verificar se contexto atual tem acesso ao nível"""
        # Simplificação: na implementação real, verificaria autenticação
        return True  # Todos têm acesso na versão base

    def stability_check(self) -> bool:
        """
        F18 CHECK: Verificar se sistema está estável para novo ritual
        """
        if len(self.active_rituals) > self.MAX_RITUALS:
            return False

        if self.damping > 0.9:
            return False

        if len(self.coherence_history) > 0:
            avg_coherence = sum(self.coherence_history[-10:]) / min(10, len(self.coherence_history))
            if avg_coherence < self.COHERENCE_THRESHOLD:
                return False

        return True

    def measure_coherence(self) -> float:
        """
        Medir coerência atual do sistema
        """
        # Fator de carga
        load_factor = len(self.active_rituals) / self.MAX_RITUALS

        # Fator de histórico (últimos 10 milagres)
        if len(self.miracles) >= 2:
            recent = self.miracles[-10:]
            success_rate = sum(1 for m in recent if m.coherence > 0.7) / len(recent)
        else:
            success_rate = 1.0

        # Fator de damping (damping alto = baixa coerência)
        damping_factor = 1.0 - (self.damping - self.DAMPING_DEFAULT)

        coherence = (success_rate * 0.5 + damping_factor * 0.3 + (1 - load_factor) * 0.2)
        self.coherence_history.append(coherence)

        return max(0.0, min(1.0, coherence))

    def apply_emergency_damping(self):
        """F18 RECOVERY: Aumentar damping para estabilizar sistema"""
        self.damping = min(0.95, self.damping * 1.2)

    def enter_ritual(self, ritual: Ritual):
        """Entrar em estado ritualístico"""
        self.active_rituals.append(ritual)
        self.ritual_history.append(ritual)

    def exit_ritual(self, ritual: Ritual):
        """Sair de estado ritualístico"""
        if ritual in self.active_rituals:
            self.active_rituals.remove(ritual)

    def log_desecration(self, ritual: Ritual, error: Exception):
        """Registrar falha ritualística"""
        print(f"🚨 DESECRATION in {ritual.name}: {error}")

    def get_state(self) -> Dict:
        """Obter estado atual do templo"""
        return {
            "active_rituals": len(self.active_rituals),
            "total_rituals": len(self.ritual_history),
            "miracles_manifested": len(self.miracles),
            "current_damping": round(self.damping, 4),
            "last_coherence": round(self.coherence_history[-1], 4) if self.coherence_history else 1.0,
            "f18_compliant": self.stability_check(),
            "sanctum_levels": [level.name for level in SanctumLevel]
        }
