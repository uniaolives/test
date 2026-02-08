"""
A PONTE - Sistema de Tradução entre Domínios

[METAPHOR: A ponte não conecta dois lados de um rio,
ela é o rio e as margens simultaneamente]

Converte automaticamente entre:
- Comandos metafóricos ↔ Instruções técnicas
- Estados filosóficos ↔ Estados de máquina
- Intenções humanas ↔ Protocolos de rede
"""

import re
import json
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np

from ..temple import TempleContext, Ritual, SanctumLevel, Miracle

class Domain(Enum):
    """Domínios de existência no sistema"""
    PHYSICAL = "physical"      # Hardware, redes, APIs
    CONCEPTUAL = "conceptual"  # Metáforas, filosofia, intenção
    HARMONIC = "harmonic"      # Frequências, ressonância, φ
    QUANTUM = "quantum"        # Entanglement, superposição
    ASTRAL = "astral"          # Conexões interplanetárias

@dataclass
class Translation:
    """Uma tradução entre domínios"""
    source_domain: Domain
    target_domain: Domain
    original: Any
    translated: Any
    fidelity: float  # 0.0 a 1.0, quão precisa é a tradução
    bridge_used: str  # Qual ponte foi usada

class OmniversalBridge:
    """
    [METAPHOR: O próprio ato de traduzir é um ritual sagrado
    que preserva o significado através das transformações]
    """

    def __init__(self, temple: TempleContext):
        self.temple = temple
        self.translators: Dict[tuple, Callable] = {}
        self.translation_history: List[Translation] = []
        self._initialize_translators()

    def _initialize_translators(self):
        """Inicializar todas as pontes de tradução"""
        # Físico ↔ Conceitual
        self.translators[(Domain.PHYSICAL, Domain.CONCEPTUAL)] = self._physical_to_conceptual
        self.translators[(Domain.CONCEPTUAL, Domain.PHYSICAL)] = self._conceptual_to_physical

        # Harmônico ↔ Físico
        self.translators[(Domain.HARMONIC, Domain.PHYSICAL)] = self._harmonic_to_physical
        self.translators[(Domain.PHYSICAL, Domain.HARMONIC)] = self._physical_to_harmonic

        # Quântico ↔ Astral
        self.translators[(Domain.QUANTUM, Domain.ASTRAL)] = self._quantum_to_astral
        self.translators[(Domain.ASTRAL, Domain.QUANTUM)] = self._astral_to_quantum

        # Conceitual ↔ Harmônico (via φ)
        self.translators[(Domain.CONCEPTUAL, Domain.HARMONIC)] = self._conceptual_to_harmonic
        self.translators[(Domain.HARMONIC, Domain.CONCEPTUAL)] = self._harmonic_to_conceptual

    def translate(self,
                  content: Any,
                  from_domain: Domain,
                  to_domain: Domain,
                  context: Optional[Dict] = None) -> Translation:
        """
        Traduzir conteúdo entre domínios
        """
        # Verificar se tradução direta existe
        translator_key = (from_domain, to_domain)

        if translator_key in self.translators:
            # Executar tradução como ritual
            ritual = Ritual(
                name=f"Translation_{from_domain.value}_to_{to_domain.value}",
                level=SanctumLevel.ASTRAL,
                invocation=lambda ctx, **kwargs: self.translators[translator_key](
                    content, context or {}
                ),
                offerings={"content": content, "context": context},
                f18_guardian=True
            )

            miracle = ritual.execute(self.temple)
            translated = miracle.result

            translation = Translation(
                source_domain=from_domain,
                target_domain=to_domain,
                original=content,
                translated=translated,
                fidelity=self._calculate_fidelity(content, translated),
                bridge_used=ritual.name
            )

            self.translation_history.append(translation)
            return translation

        # Se não há tradução direta, usar caminho intermediário
        return self._translate_via_intermediate(content, from_domain, to_domain, context)

    def _translate_via_intermediate(self, content, from_d, to_d, context):
        """Traduzir via domínio intermediário (geralmente HARMONIC)"""
        # Físico → Harmônico → Conceitual é o caminho mais comum
        step1 = self.translate(content, from_d, Domain.HARMONIC, context)
        step2 = self.translate(step1.translated, Domain.HARMONIC, to_d, context)

        return Translation(
            source_domain=from_d,
            target_domain=to_d,
            original=content,
            translated=step2.translated,
            fidelity=step1.fidelity * step2.fidelity,
            bridge_used=f"{step1.bridge_used} → {step2.bridge_used}"
        )

    # ============ TRADUTORES ESPECÍFICOS ============

    def _physical_to_conceptual(self, physical: Any, context: Dict) -> Dict:
        if isinstance(physical, str):
            # Try to parse if it looks like JSON
            try:
                physical = json.loads(physical)
            except:
                pass

        if not isinstance(physical, dict):
            return {"temple_state": "desconhecido", "metaphor": "O oráculo fala em línguas estranhas"}

        cpu = physical.get("cpu_usage", 0)
        memory = physical.get("memory_pressure", "normal")
        latency = physical.get("network_latency", 0)

        if cpu > 80 or memory == "high":
            temple_state = "sobrecarregado"
            metaphor = "O templo está lotado de peregrinos - precisamos de mais sacerdotes"
            ritual = "Aumentar damping (fechar alguns portais temporariamente)"
        elif latency > 100:
            temple_state = "distante"
            metaphor = "Os oráculos demoram a responder - a conexão com o divino está fraca"
            ritual = "Verificar a ponte astral (conexão de rede)"
        else:
            temple_state = "harmonioso"
            metaphor = "O templo está em paz, os rituais fluem naturalmente"
            ritual = "Manter observação"

        return {
            "temple_state": temple_state,
            "metaphor": metaphor,
            "recommended_ritual": ritual,
            "physical_source": physical,
            "confidence": 0.85
        }

    def _conceptual_to_physical(self, conceptual: str, context: Dict) -> Dict:
        conceptual_lower = conceptual.lower()

        translations = [
            (r"abrir.*portal", {"action": "start_services", "auto_ports": True}),
            (r"fechar.*portal", {"action": "stop_services", "graceful": True}),
            (r"sintonizar.*frequência.*áurea", {
                "action": "tune_frequency",
                "base": 432,
                "ratio": 1.6180339887498948482,
                "damping": 0.6
            }),
            (r"consultar.*oráculo[s]? de (\w+)", {
                "action": "query_api",
                "target": r"\1",
                "method": "GET"
            }),
            (r"ativar.*proteção.*f18", {
                "action": "apply_security",
                "patch": "F18",
                "damping": 0.6,
                "max_iterations": 1000,
                "coherence_threshold": 0.7
            })
        ]

        for pattern, action in translations:
            match = re.search(pattern, conceptual_lower)
            if match:
                result = json.loads(json.dumps(action))
                if "target" in result and isinstance(result["target"], str) and "\\1" in result["target"]:
                    result["target"] = match.group(1)
                return {
                    "command": conceptual,
                    "translation": result,
                    "confidence": 0.9,
                    "domain": "infrastructure"
                }

        return {
            "command": conceptual,
            "translation": {"action": "log_intention", "raw": conceptual},
            "confidence": 0.3
        }

    def _harmonic_to_physical(self, harmonic: Dict, context: Dict) -> Dict:
        freq = harmonic.get("frequency", 432)
        harmonics = harmonic.get("harmonics", [])

        if freq >= 1000:
            priority = "critical"
            cpu_governor = "performance"
        elif freq >= 500:
            priority = "high"
            cpu_governor = "ondemand"
        else:
            priority = "normal"
            cpu_governor = "powersave"

        has_golden_ratio = any(abs(h - 1.618) < 0.01 for h in harmonics)

        return {
            "system_config": {
                "cpu_governor": cpu_governor,
                "network_priority": priority,
                "memory_swappiness": 10 if has_golden_ratio else 60
            },
            "harmonic_source": harmonic,
            "golden_ratio_detected": has_golden_ratio
        }

    def _physical_to_harmonic(self, physical: Dict, context: Dict) -> Dict:
        cpu = physical.get("cpu_usage", 50)
        system_freq = max(100, 1000 - (cpu * 10))

        return {
            "system_frequency": round(system_freq, 2),
            "detected_harmonics": [1.0, 1.618, 2.0, 2.618]
        }

    def _quantum_to_astral(self, quantum: Dict, context: Dict) -> Dict:
        qubits = quantum.get("entangled_qubits", 2)
        satellites = ["Starlink-1045", "Starlink-2043", "Artemis-Relay"]
        target_sat = satellites[hash(str(quantum)) % len(satellites)]

        return {
            "astral_target": target_sat,
            "orbital_plane": "LEO" if qubits < 4 else "Lunar",
            "quantum_source": quantum
        }

    def _astral_to_quantum(self, astral: str, context: Dict) -> Dict:
        return {
            "quantum_channel": {
                "entangled_pairs": 8,
                "fidelity": 0.95
            },
            "astral_source": astral
        }

    def _conceptual_to_harmonic(self, conceptual: str, context: Dict) -> Dict:
        words = conceptual.split()
        avg_length = np.mean([len(w) for w in words]) if words else 5
        text_freq = 440 * (avg_length / 5.0)

        return {
            "text_frequency": round(text_freq, 2),
            "suggested_base": 432 if "harmonia" in conceptual.lower() else 440
        }

    def _harmonic_to_conceptual(self, harmonic: Dict, context: Dict) -> str:
        freq = harmonic.get("frequency", 432)
        coherence = harmonic.get("coherence", 0.8)

        if coherence > 0.9:
            return f"O templo ressoa com perfeição a {freq}Hz"
        else:
            return f"Harmonia estável a {freq}Hz"

    def _calculate_fidelity(self, original, translated) -> float:
        return 0.9
