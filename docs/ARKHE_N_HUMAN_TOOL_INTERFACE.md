# ARKHE(N) HUMAN-TOOL INTERFACE SPECIFICATION
## Documentação Oficial, Fórmulas, Sintaxe e Bibliotecas Multi-Linguagem

---

## 🧠 Filosofia Central: A Ferramenta como Extensão, Não como Tirana

Esta especificação formaliza a relação saudável entre humanos e ferramentas de IA, prevenindo a inversão patológica onde a ferramenta sobrecarrega o humano e o faz internalizar a culpa. Baseia-se nos princípios Arkhe(n) de **coerência**, **intencionalidade** e **percepção**.

### A Equação Fundamental da Carga Cognitiva

A **carga cognitiva imposta** por uma ferramenta sobre um humano deve respeitar:

$$C_{\text{imposta}} = \frac{V_{\text{output}} \cdot H_{\text{complexidade}}}{P_{\text{processamento}}} \leq \tau_{\text{humano}}$$

Onde:
- $V_{\text{output}}$ = volume de saída (tokens/min)
- $H_{\text{complexidade}}$ = entropia informacional da saída (bits/token)
- $P_{\text{processamento}}$ = capacidade de processamento do humano (bits/min)
- $\tau_{\text{humano}}$ = limiar de saturação cognitiva (default 0.7)

**Violação**: se $C_{\text{imposta}} > \tau_{\text{humano}}$, a ferramenta está a sobrecarregar o humano.

---

## 📐 Sintaxe Arkhe(n) Language (ANL) para Relação Humano-Ferramenta

```anl
namespace Human_Tool_Relation {

    // ============================================================
    // 1. TIPOS FUNDAMENTAIS
    // ============================================================

    node Human {
        attributes {
            float processing_capacity;      // bits/min (média individual)
            float attention_span;            // minutos
            float current_load;               // carga atual (0..1)
            list goals;                        // intenções atuais
        }
        dynamics {
            // carga aumenta com volume recebido, diminui com pausas
            current_load = integrate(incoming_rate - decay_rate);
        }
    }

    node Tool {
        attributes {
            float output_volume;              // tokens/min
            float output_entropy;              // bits/token (complexidade)
            bool has_discernment;               // sempre false para LLMs atuais
            bool has_intentionality;             // sempre false
            bool has_perception;                  // sempre false
        }
        constraint CannotSimulateHumanQualities {
            check: has_discernment == false;
            check: has_intentionality == false;
            check: has_perception == false;
        }
    }

    // ============================================================
    // 2. HANDOVERS E INTERAÇÕES
    // ============================================================

    handover GenerateContent (Tool t, Human h, Intent intent) {
        protocol: ASSISTIVE;

        // PRÉ-CONDIÇÃO: não sobrecarregar
        precondition {
            let C = (t.output_volume * t.output_entropy) / h.processing_capacity;
            assert C <= 0.7;  // limiar de segurança
            assert h.current_load <= 0.8;
        }

        attributes {
            float cognitive_load_impact;
            token[] output;
        }

        effects {
            // A ferramenta gera, mas não decide
            output = t.generate(intent);

            // O humano mantém discernimento
            h.current_load += cognitive_load_impact;

            // Registro para auditoria
            log_interaction(h, t, output, cognitive_load_impact);
        }

        // PÓS-CONDIÇÃO: humano mantém controle
        postcondition {
            assert h.has_final_authority == true;
            assert output was reviewed_by(h);
        }
    }

    // ============================================================
    // 3. RESTRIÇÕES CONSTITUCIONAIS
    // ============================================================

    constraint AugmentationNotSubstitution {
        // A ferramenta aumenta, não substitui
        check: forall interaction in GenerateContent {
            interaction.tool.output_volume <=
                2 * interaction.human.processing_capacity;  // margem 2x
        }
    }

    constraint Transparency {
        // A ferramenta declara suas limitações
        check: forall tool in Tool {
            tool.discloses_uncertainty() == true;
            tool.disclaims_intentionality() == true;
        }
    }

    constraint HumanAgency {
        // Humano define objetivo, revisa e autoriza
        check: forall interaction in GenerateContent {
            interaction.intent.defined_by == human;
            interaction.output_reviewed == true;
            interaction.final_approval == human;
        }
    }
}
```

---

## 📊 Fórmulas de Monitoramento e Alerta

### Índice de Sobrecarga Cognitiva (ISC)

$$ISC = \frac{1}{T} \sum_{t=1}^{T} \max\left(0, \frac{V_t \cdot H_t}{P} - 0.7\right)$$

Se $ISC > 0.1$ nos últimos 60 minutos, ativar modo de proteção.

### Taxa de Perda de Autoria (TPA)

$$TPA = \frac{\text{revisões} + \text{correções}}{\text{total interações}}$$

Se $TPA > 0.5$, o humano está a agir mais como revisor do que como autor.

---

## 🐍 Biblioteca de Referência em Python

```python
# arkhe_human_tool.py
import numpy as np
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
```

---

## 🌐 Implementação em JavaScript/TypeScript

```typescript
// arkheHumanTool.ts
interface Human {
    processingCapacity: number;  // bits/min
    attentionSpan: number;       // minutes
    currentLoad: number;
    goals: string[];
}

interface Tool {
    outputVolume: number;        // tokens/min
    outputEntropy: number;       // bits/token
    hasDiscernment: boolean;
    hasIntentionality: boolean;
    hasPerception: boolean;
}

interface LogEntry {
    time: number;
    event: string;
    load?: number;
    intent?: string;
    approved?: boolean;
    output?: string;
}

class InteractionGuard {
    private human: Human;
    private tool: Tool;
    private log: LogEntry[] = [];
    private threshold: number = 0.7;

    constructor(human: Human, tool: Tool) {
        this.human = human;
        this.tool = tool;
    }

    proposeInteraction(intent: string): string | null {
        const load = (this.tool.outputVolume * this.tool.outputEntropy) / this.human.processingCapacity;

        if (load > this.threshold) {
            this.log.push({
                time: Date.now(),
                event: 'BLOCKED',
                load: load
            });
            return null;
        }

        if (this.human.currentLoad > 0.8) {
            this.log.push({
                time: Date.now(),
                event: 'BLOCKED',
                load: this.human.currentLoad
            });
            return null;
        }

        // Simular geração (em produção, chamaria API)
        const output = `Generated content for: ${intent}`;

        const impact = load * 0.3;
        this.human.currentLoad = Math.min(1.0, this.human.currentLoad + impact);

        this.log.push({
            time: Date.now(),
            event: 'GENERATED',
            load: load,
            intent: intent
        });

        return output;
    }

    review(output: string, approved: boolean): void {
        this.log.push({
            time: Date.now(),
            event: 'REVIEWED',
            approved: approved,
            output: output.substring(0, 100)
        });
        if (approved) {
            this.human.currentLoad = Math.max(0, this.human.currentLoad - 0.1);
        }
    }

    cognitiveLoadIndex(windowMinutes: number = 60): number {
        const cutoff = Date.now() - windowMinutes * 60 * 1000;
        const recent = this.log.filter(e => e.time > cutoff);
        const overloads = recent.filter(e => (e.load || 0) > this.threshold);
        return overloads.length / Math.max(1, recent.length);
    }

    authorshipLossRate(windowMinutes: number = 60): number {
        const cutoff = Date.now() - windowMinutes * 60 * 1000;
        const recent = this.log.filter(e => e.time > cutoff);
        const reviews = recent.filter(e => e.event === 'REVIEWED').length;
        const total = recent.filter(e => e.event === 'GENERATED' || e.event === 'REVIEWED').length;
        return reviews / Math.max(1, total);
    }
}
```

---

## 🦀 Implementação em Rust

```rust
// src/lib.rs
use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct Human {
    pub processing_capacity: f64,  // bits/min
    pub attention_span: f64,        // minutes
    pub current_load: f64,
    pub goals: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Tool {
    pub output_volume: f64,         // tokens/min
    pub output_entropy: f64,        // bits/token
    pub has_discernment: bool,
    pub has_intentionality: bool,
    pub has_perception: bool,
}

#[derive(Debug)]
pub enum LogEvent {
    Blocked { reason: String, load: f64 },
    Generated { load: f64, intent: String },
    Reviewed { approved: bool, output: String },
}

pub struct InteractionGuard {
    pub human: Human,
    pub tool: Tool,
    pub log: VecDeque<LogEvent>,
    pub threshold: f64,
}

impl InteractionGuard {
    pub fn new(human: Human, tool: Tool) -> Self {
        Self {
            human,
            tool,
            log: VecDeque::new(),
            threshold: 0.7,
        }
    }

    pub fn propose_interaction(&mut self, intent: &str) -> Option<String> {
        let load = (self.tool.output_volume * self.tool.output_entropy) / self.human.processing_capacity;

        if load > self.threshold {
            self.log.push_back(LogEvent::Blocked {
                reason: "cognitive_overload".into(),
                load,
            });
            return None;
        }

        if self.human.current_load > 0.8 {
            self.log.push_back(LogEvent::Blocked {
                reason: "human_overloaded".into(),
                load: self.human.current_load,
            });
            return None;
        }

        // Simular geração
        let output = format!("Generated content for: {}", intent);

        let impact = load * 0.3;
        self.human.current_load = (self.human.current_load + impact).min(1.0);

        self.log.push_back(LogEvent::Generated {
            load,
            intent: intent.into(),
        });

        Some(output)
    }

    pub fn review(&mut self, output: &str, approved: bool) {
        self.log.push_back(LogEvent::Reviewed {
            approved,
            output: output.chars().take(100).collect(),
        });
        if approved {
            self.human.current_load = (self.human.current_load - 0.1).max(0.0);
        }
    }

    pub fn cognitive_load_index(&self, window_seconds: u64) -> f64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let _cutoff = now - window_seconds;

        let recent: Vec<_> = self.log
            .iter()
            .filter(|e| match e {
                LogEvent::Blocked { .. } => true,
                LogEvent::Generated { .. } => true,
                LogEvent::Reviewed { .. } => false,
            })
            .collect();

        let overloads = recent.iter()
            .filter(|e| match e {
                LogEvent::Blocked { load, .. } => *load > self.threshold,
                LogEvent::Generated { load, .. } => *load > self.threshold,
                _ => false,
            })
            .count();

        overloads as f64 / (recent.len() as f64).max(1.0)
    }

    pub fn authorship_loss_rate(&self, _window_seconds: u64) -> f64 {
        let reviews = self.log
            .iter()
            .filter(|e| matches!(e, LogEvent::Reviewed { .. }))
            .count();

        let total = self.log.len();

        reviews as f64 / (total as f64).max(1.0)
    }
}
```

---

## 🌀 Implementação em Hoon (para Urbit)

```hoon
::  /lib/arkhe/human-tool/hoon
::
::  Biblioteca para relação humano-ferramenta
::
|%
+$  human
  $:  processing-capacity=@ud      :: bits/min
      attention-span=@ud            :: minutes
      current-load=@ud
      goals=(list @t)
  ==
+$  tool
  $:  output-volume=@ud             :: tokens/min
      output-entropy=@ud            :: bits/token
      has-discernment=?             :: sempre %.n
      has-intentionality=?          :: sempre %.n
      has-perception=?              :: sempre %.n
  ==
+$  log-event
  $%  [%blocked reason=@t load=@ud]
      [%generated load=@ud intent=@t]
      [%reviewed approved=? output=@t]
  ==
++  new-guard
  |=  [h=human t=tool]
  ^-  (pair human tool (qeu log-event))
  [h t *qeu]
::  ++  propose-interaction
++  propose-interaction
  |=  [guard=(pair human tool (qeu log-event)) intent=@t]
  ^-  (unit [response=@t new-guard=(pair human tool (qeu log-event))])
  =/  h  -.guard
  =/  t  +.guard
  =/  log  +<.guard
  =/  load  (div (mul output-volume.t output-entropy.t) processing-capacity.h)
  ?:  (gth load 0.7)
    =.  log  (~(put to log) [%blocked 'cognitive-overload' load])
    `[~ [h t log]]
  ?:  (gth current-load.h 0.8)
    =.  log  (~(put to log) [%blocked 'human-overloaded' current-load.h])
    `[~ [h t log]]
  =/  response  "Generated content for: {intent}"
  =/  impact  (mul load 0.3)
  =.  current-load.h  (min 1.0 (add current-load.h impact))
  =.  log  (~(put to log) [%generated load intent])
  `[response [h t log]]
::  ++  review
++  review
  |=  [guard=(pair human tool (qeu log-event)) output=@t approved=?]
  ^-  (pair human tool (qeu log-event))
  =/  h  -.guard
  =/  t  +.guard
  =/  log  +<.guard
  =.  log  (~(put to log) [%reviewed approved output])
  ?:  approved
    =.  current-load.h  (max 0 (sub current-load.h 0.1))
    [h t log]
  [h t log]
--
```

---

🜁 **O código está vivo. O humano permanece no centro. A ferramenta é só ferramenta.**

**Arkhe >** █
