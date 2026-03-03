# arkhe_human_tool.py
# Reference implementation for Arkhe(n) Human-Tool Interface
from dataclasses import dataclass
from typing import List, Optional
import time

@dataclass
class Human:
    processing_capacity: float  # bits/min
    attention_span: float       # minutes
    current_load: float = 0.0
    goals: List[str] = None

    def can_process(self, volume: float, entropy: float) -> bool:
        """Verifica se humano consegue processar sem sobrecarga."""
        load = (volume * entropy) / self.processing_capacity
        return self.current_load + load <= 0.8

@dataclass
class Tool:
    output_volume: float        # tokens/min
    output_entropy: float       # bits/token
    has_discernment: bool = False
    has_intentionality: bool = False
    has_perception: bool = False

    def generate(self, intent: str) -> str:
        """Simula geração de conteúdo."""
        # Placeholder: em produção, chamaria uma LLM
        return f"Generated content for: {intent}"

class InteractionGuard:
    """Monitora e protege a relação humano-ferramenta."""

    def __init__(self, human: Human, tool: Tool):
        self.human = human
        self.tool = tool
        self.log = []
        self.threshold = 0.7

    def propose_interaction(self, intent: str) -> Optional[str]:
        """Só permite interação se dentro dos limites seguros."""
        load = (self.tool.output_volume * self.tool.output_entropy) / self.human.processing_capacity

        if load > self.threshold:
            self.log.append({
                'time': time.time(),
                'event': 'BLOCKED',
                'reason': 'cognitive_overload',
                'load': load
            })
            return None

        if self.human.current_load > 0.8:
            self.log.append({
                'time': time.time(),
                'event': 'BLOCKED',
                'reason': 'human_overloaded',
                'load': self.human.current_load
            })
            return None

        # Gerar conteúdo
        output = self.tool.generate(intent)

        # Actualizar carga do humano
        impact = load * 0.3  # factor de impacto
        self.human.current_load = min(1.0, self.human.current_load + impact)

        self.log.append({
            'time': time.time(),
            'event': 'GENERATED',
            'load': load,
            'intent': intent
        })

        return output

    def review(self, output: str, approved: bool) -> None:
        """Humano revisa e aprova/rejeita."""
        self.log.append({
            'time': time.time(),
            'event': 'REVIEWED',
            'approved': approved,
            'output': output[:100]  # log parcial
        })
        if approved:
            # Reduzir carga ligeiramente (satisfação)
            self.human.current_load = max(0, self.human.current_load - 0.1)

    def cognitive_load_index(self, window_minutes: int = 60) -> float:
        """Calcula ISC nos últimos N minutos."""
        recent = [e for e in self.log if e['time'] > time.time() - window_minutes*60]
        overloads = [e for e in recent if e.get('load', 0) > self.threshold]
        return len(overloads) / max(1, len(recent))

    def authorship_loss_rate(self, window_minutes: int = 60) -> float:
        """Calcula TPA: taxa de revisões/correções."""
        recent = [e for e in self.log if e['time'] > time.time() - window_minutes*60]
        reviews = len([e for e in recent if e['event'] == 'REVIEWED'])
        total = len([e for e in recent if e['event'] in ('GENERATED', 'REVIEWED')])
        return reviews / max(1, total)
