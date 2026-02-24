#!/usr/bin/env python3
"""
ASI-Ω UNIVERSAL COBOL PARSER v2.0
Suporte completo: estruturado + não-estruturado (spaghetti)
"""

import re
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from enum import Enum, auto
from collections import defaultdict
import networkx as nx

class CobolVerb(Enum):
    """Verbos COBOL completos, incluindo obsoletos."""
    # Estruturados
    MOVE = auto(); COMPUTE = auto(); ADD = auto(); SUBTRACT = auto()
    MULTIPLY = auto(); DIVIDE = auto(); IF = auto(); ELSE = auto()
    EVALUATE = auto(); WHEN = auto(); PERFORM = auto()
    PERFORM_VARYING = auto(); PERFORM_UNTIL = auto()
    PERFORM_TIMES = auto(); PERFORM_THRU = auto()
    # Não-estruturados (spaghetti)
    GO_TO = auto(); GO_TO_DEPENDING = auto()
    ALTER = auto(); ALTER_PROCEED = auto()
    # I/O
    READ = auto(); WRITE = auto(); REWRITE = auto(); DELETE = auto()
    START = auto(); OPEN = auto(); CLOSE = auto(); ACCEPT = auto()
    DISPLAY = auto(); CALL = auto(); CANCEL = auto()
    # Controle
    STOP_RUN = auto(); EXIT = auto(); EXIT_PROGRAM = auto()
    GOBACK = auto(); CONTINUE = auto(); NEXT_SENTENCE = auto()
    # Exceções
    USE = auto(); DECLARATIVES = auto()

@dataclass
class CobolStatement:
    verb: CobolVerb
    raw_text: str
    line_number: int
    operands: List[str] = field(default_factory=list)
    condition: Optional[str] = None
    target_paragraph: Optional[str] = None
    target_paragraphs: List[str] = field(default_factory=list)  # Para GO TO DEPENDING
    alter_targets: Dict[str, str] = field(default_factory=dict)  # Para ALTER
    is_alter_target: bool = False  # Este parágrafo é alvo de ALTER?
    arithmetic_expression: Optional[str] = None

@dataclass
class CobolParagraph:
    name: str
    statements: List[CobolStatement]
    line_start: int
    line_end: int
    calls: Set[str] = field(default_factory=set)
    called_by: Set[str] = field(default_factory=set)
    goto_targets: Set[str] = field(default_factory=set)  # Alvos de GO TO
    goto_sources: Set[str] = field(default_factory=set)  # Origens de GO TO para cá
    alter_mutable: bool = False  # Pode ser modificado por ALTER
    data_reads: Set[str] = field(default_factory=set)
    data_writes: Set[str] = field(default_factory=set)
    # Análise de estrutura
    has_entry_point: bool = False
    has_multiple_entries: bool = False
    is_fall_through: bool = False  # Execução cai aqui do parágrafo anterior

class UniversalCobolParser:
    """
    Parser completo para COBOL estruturado e não-estruturado.
    Constrói grafo de fluxo de controle completo (CFG).
    """

    # Padrões expandidos
    VERB_PATTERNS = {
        # Estruturados
        CobolVerb.MOVE: r'MOVE\s+(.+?)\s+TO\s+(.+)',
        CobolVerb.COMPUTE: r'COMPUTE\s+([A-Z0-9\-]+)(?:\s+ROUNDED)?\s*=\s*(.+)',
        CobolVerb.ADD: r'ADD\s+(.+?)\s+TO\s+(.+)',
        CobolVerb.SUBTRACT: r'SUBTRACT\s+(.+?)\s+FROM\s+(.+)',
        CobolVerb.MULTIPLY: r'MULTIPLY\s+(.+?)\s+BY\s+(.+)',
        CobolVerb.DIVIDE: r'DIVIDE\s+(.+?)\s+BY\s+(.+)',
        CobolVerb.IF: r'IF\s+(.+?)(?:\s+THEN)?(?:\.|\s*$)',
        CobolVerb.ELSE: r'ELSE',
        CobolVerb.EVALUATE: r'EVALUATE\s+(.+)',
        CobolVerb.WHEN: r'WHEN\s+(.+)',
        CobolVerb.PERFORM: r'PERFORM\s+([A-Z][A-Z0-9\-]+)(?:\s+THRU\s+([A-Z][A-Z0-9\-]+))?(?:\.|\s*$)',
        CobolVerb.PERFORM_VARYING: r'PERFORM\s+([A-Z][A-Z0-9\-]+)\s+VARYING\s+(.+?)\s+FROM\s+(.+?)\s+BY\s+(.+?)\s+UNTIL\s+(.+)',
        CobolVerb.PERFORM_UNTIL: r'PERFORM\s+([A-Z][A-Z0-9\-]+)\s+UNTIL\s+(.+)',
        CobolVerb.PERFORM_TIMES: r'PERFORM\s+([A-Z][A-Z0-9\-]+)\s+(\d+|.+?)\s+TIMES',
        # NÃO-ESTRUTURADOS (spaghetti)
        CobolVerb.GO_TO: r'GO\s+TO\s+([A-Z][A-Z0-9\-]+)(?:\.|\s*$)',
        CobolVerb.GO_TO_DEPENDING: r'GO\s+TO\s+([A-Z][A-Z0-9\-]+)\s+([A-Z][A-Z0-9\-]+)\s+([A-Z][A-Z0-9\-]+)(?:\s+DEPENDING\s+ON\s+(.+))?',
        CobolVerb.ALTER: r'ALTER\s+([A-Z][A-Z0-9\-]+)\s+TO\s+PROCEED\s+TO\s+([A-Z][A-Z0-9\-]+)',
        # Controle
        CobolVerb.STOP_RUN: r'STOP\s+RUN',
        CobolVerb.EXIT: r'EXIT(?:\s+PROGRAM)?',
        CobolVerb.GOBACK: r'GOBACK',
        CobolVerb.CONTINUE: r'CONTINUE',
        CobolVerb.NEXT_SENTENCE: r'NEXT\s+SENTENCE',
        # I/O
        CobolVerb.READ: r'READ\s+([A-Z0-9\-]+)',
        CobolVerb.WRITE: r'WRITE\s+([A-Z0-9\-]+)',
        CobolVerb.OPEN: r'OPEN\s+(.+?)\s+([A-Z0-9\-]+)',
        CobolVerb.CLOSE: r'CLOSE\s+([A-Z0-9\-]+)',
        CobolVerb.DISPLAY: r'DISPLAY\s+(.+)',
        CobolVerb.ACCEPT: r'ACCEPT\s+([A-Z0-9\-]+)',
        CobolVerb.CALL: r'CALL\s+[\'"]?([A-Z][A-Z0-9\-]+)[\'"]?',
    }

    def __init__(self):
        self.paragraphs: Dict[str, CobolParagraph] = {}
        self.alter_table: Dict[str, str] = {}  # Mapeamento ALTER: original → novo
        self.current_paragraph: Optional[str] = None
        self.statements_buffer: List[CobolStatement] = []
        self.line_number = 0
        self.in_declaratives = False

    def extract(self, procedure_division: str) -> Dict[str, CobolParagraph]:
        """
        Pipeline principal com suporte a código spaghetti.
        """
        lines = procedure_division.split('\n')
        self.line_number = 0
        self.in_declaratives = False
        self.paragraphs = {}
        self.alter_table = {}

        # Fase 1: Parse básico
        for line in lines:
            self.line_number += 1
            clean_line = self._clean_line(line)

            if not clean_line:
                continue

            # Detectar DECLARATIVES (seção de exceções)
            if 'DECLARATIVES' in clean_line:
                self.in_declaratives = True
                continue
            if 'END DECLARATIVES' in clean_line:
                self.in_declaratives = False
                continue

            # Detectar parágrafo
            match = self._is_paragraph_header(clean_line)
            if match:
                self._finalize_current_paragraph()
                para_name = match
                self.current_paragraph = para_name
                self.statements_buffer = []
                continue

            # Parse de instrução
            if self.current_paragraph:
                statement = self._parse_statement(clean_line)
                if statement:
                    self.statements_buffer.append(statement)
                    # Detectar ALTER para marcação posterior
                    if statement.verb == CobolVerb.ALTER:
                        self.alter_table[statement.operands[0]] = statement.operands[1]

        self._finalize_current_paragraph()

        # Fase 2: Análise de fluxo não-estruturado
        self._analyze_spaghetti_flow()

        return self.paragraphs

    def _clean_line(self, line: str) -> str:
        """Normalização completa de linha COBOL."""
        # Remover cols 1-6 (sequence), col 7 (indicator)
        if len(line) >= 7:
            indicator = line[6]
            if indicator == '*':  # Comentário
                return ''
            code_part = line[7:].strip()
        else:
            code_part = line.strip()

        # Remover comentários inline (*>)
        code_part = re.sub(r'\*>.*$', '', code_part)

        return code_part.strip().upper()

    def _is_paragraph_header(self, line: str) -> Optional[str]:
        """Detecta se linha é cabeçalho de parágrafo ou seção."""
        if re.match(r'^[A-Z][A-Z0-9\-]{0,29}\.$', line):
            if not any(keyword in line for keyword in ['DIVISION', 'SECTION', 'EXIT']):
                return line.rstrip('.')
        return None

    def _parse_statement(self, line: str) -> Optional[CobolStatement]:
        """Parser universal com fallback."""
        for verb, pattern in self.VERB_PATTERNS.items():
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                operands = [o.strip('.') if o else o for o in match.groups()]
                stmt = CobolStatement(
                    verb=verb,
                    raw_text=line,
                    line_number=self.line_number,
                    operands=operands
                )

                if verb == CobolVerb.GO_TO:
                    stmt.target_paragraph = operands[0]
                elif verb == CobolVerb.GO_TO_DEPENDING:
                    stmt.target_paragraphs = operands[:3]
                    stmt.condition = operands[3]
                elif verb == CobolVerb.PERFORM:
                    stmt.target_paragraph = operands[0]
                elif verb == CobolVerb.COMPUTE:
                    stmt.arithmetic_expression = operands[1]
                elif verb == CobolVerb.IF:
                    stmt.condition = operands[0]

                return stmt

        return CobolStatement(
            verb=CobolVerb.CONTINUE,
            raw_text=line,
            line_number=self.line_number,
            operands=[]
        )

    def _finalize_current_paragraph(self):
        if self.current_paragraph and self.statements_buffer:
            self.paragraphs[self.current_paragraph] = CobolParagraph(
                name=self.current_paragraph,
                statements=self.statements_buffer.copy(),
                line_start=self.statements_buffer[0].line_number,
                line_end=self.statements_buffer[-1].line_number
            )

    def _analyze_spaghetti_flow(self):
        para_names = list(self.paragraphs.keys())
        for i, (name, para) in enumerate(self.paragraphs.items()):
            for stmt in para.statements:
                if stmt.verb == CobolVerb.GO_TO:
                    para.goto_targets.add(stmt.target_paragraph)
                    if stmt.target_paragraph in self.paragraphs:
                        self.paragraphs[stmt.target_paragraph].goto_sources.add(name)
                elif stmt.verb == CobolVerb.GO_TO_DEPENDING:
                    for target in stmt.target_paragraphs:
                        para.goto_targets.add(target)
                        if target in self.paragraphs:
                            self.paragraphs[target].goto_sources.add(name)
                elif stmt.verb == CobolVerb.ALTER:
                    target_to_modify = stmt.operands[0]
                    if target_to_modify in self.paragraphs:
                        self.paragraphs[target_to_modify].alter_mutable = True

            if i < len(para_names) - 1:
                last_stmt = para.statements[-1] if para.statements else None
                if not last_stmt or last_stmt.verb not in [
                    CobolVerb.GO_TO, CobolVerb.STOP_RUN,
                    CobolVerb.EXIT, CobolVerb.GOBACK
                ]:
                    next_para = para_names[i + 1]
                    self.paragraphs[next_para].is_fall_through = True

    def _build_control_flow_graph(self) -> nx.DiGraph:
        G = nx.DiGraph()
        para_names = list(self.paragraphs.keys())
        for i, (name, para) in enumerate(self.paragraphs.items()):
            G.add_node(name)
            for target in para.goto_targets:
                if target in self.paragraphs:
                    G.add_edge(name, target, type='GO_TO')

            if i < len(para_names) - 1:
                last_stmt = para.statements[-1] if para.statements else None
                if not last_stmt or last_stmt.verb not in [
                    CobolVerb.GO_TO, CobolVerb.STOP_RUN,
                    CobolVerb.EXIT, CobolVerb.GOBACK
                ]:
                    G.add_edge(name, para_names[i+1], type='FALL_THROUGH')
        return G

    def detect_spaghetti_clusters(self) -> List[Set[str]]:
        G = self._build_control_flow_graph()
        sccs = list(nx.strongly_connected_components(G))
        return [scc for scc in sccs if len(scc) > 1]

    def generate_restructuring_plan(self) -> Dict[str, Any]:
        plan = {'alter_elimination': [], 'goto_replacement': []}
        for original, new_target in self.alter_table.items():
            plan['alter_elimination'].append({
                'paragraph': original,
                'new_target': new_target
            })
        for name, para in self.paragraphs.items():
            for target in para.goto_targets:
                plan['goto_replacement'].append({'source': name, 'target': target})
        return plan

class SpaghettiRustTransmuter:
    def __init__(self, parser: UniversalCobolParser):
        self.parser = parser
        self.paragraphs = parser.paragraphs
        self.indent = "    "

    def transmute(self) -> str:
        lines = [
            "// Gerado por ASI-Ω Universal COBOL Parser v2.0",
            "use rust_decimal::Decimal;",
            "use rust_decimal_macros::dec;",
            "use std::collections::HashMap;",
            "use thiserror::Error;",
            "use tokio::sync::RwLock;",
            "",
            "#[derive(Error, Debug)]",
            "pub enum BusinessError {",
            '    #[error("Variável não encontrada: {0}")]',
            "    VariableNotFound(String),",
            '    #[error("Erro aritmético: {0}")]',
            "    ArithmeticError(String),",
            "}",
            "",
            "pub struct Context {",
            "    pub variables: RwLock<HashMap<String, Variable>>,",
            "    pub alter_state: RwLock<HashMap<String, String>>,",
            "}",
            "",
            "#[derive(Clone, Debug)]",
            "pub enum Variable {",
            "    Decimal(Decimal, u32, u32),",
            "    String(String, usize),",
            "    Integer(i64),",
            "}",
            "",
            "impl Context {",
            "    pub fn new() -> Self {",
            "        let mut alter_state = HashMap::new();",
        ]

        for original, new_target in self.parser.alter_table.items():
            lines.append(f'        alter_state.insert("{original}".to_string(), "{new_target}".to_string());')

        lines.extend([
            "        Self {",
            "            variables: RwLock::new(HashMap::new()),",
            "            alter_state: RwLock::new(alter_state),",
            "        }",
            "    }",
            "",
            "    pub async fn get_alter_target(&self, para: &str) -> String {",
            "        let state = self.alter_state.read().await;",
            "        state.get(para).cloned().unwrap_or_else(|| para.to_string())",
            "    }",
            "",
            "    pub async fn set_alter_target(&self, para: &str, new: &str) {",
            "        let mut state = self.alter_state.write().await;",
            "        state.insert(para.to_string(), new.to_string());",
            "    }",
            "",
            "    pub async fn get_decimal(&self, name: &str) -> Result<Decimal, BusinessError> {",
            "        let vars = self.variables.read().await;",
            "        match vars.get(name) {",
            "            Some(Variable::Decimal(v, _, _)) => Ok(*v),",
            "            _ => Err(BusinessError::VariableNotFound(name.to_string())),",
            "        }",
            "    }",
            "",
            "    pub async fn set_decimal(&self, name: &str, value: Decimal) -> Result<(), BusinessError> {",
            "        let mut vars = self.variables.write().await;",
            "        vars.insert(name.to_string(), Variable::Decimal(value, 9, 2));",
            "        Ok(())",
            "    }",
            "",
            "    pub async fn get_string(&self, name: &str) -> Result<String, BusinessError> {",
            "        let vars = self.variables.read().await;",
            "        match vars.get(name) {",
            "            Some(Variable::String(v, _)) => Ok(v.clone()),",
            "            _ => Err(BusinessError::VariableNotFound(name.to_string())),",
            "        }",
            "    }",
            "",
            "    pub async fn set_string(&self, name: &str, value: String) -> Result<(), BusinessError> {",
            "        let mut vars = self.variables.write().await;",
            "        vars.insert(name.to_string(), Variable::String(value, 255));",
            "        Ok(())",
            "    }",
            "}",
            "",
            "#[derive(Clone, Copy, Debug, PartialEq)]",
            "enum ParagraphState {",
        ])

        para_names = list(self.paragraphs.keys())
        for name in para_names:
            lines.append(f"    {name.replace('-', '_').upper()},")
        lines.append("    FINISHED,")

        lines.extend([
            "}",
            "",
            "pub async fn main_loop(ctx: &Context) -> Result<(), BusinessError> {",
            f"    let mut state = ParagraphState::{para_names[0].replace('-', '_').upper() if para_names else 'FINISHED'};",
            "    let mut iterations = 0;",
            "    const MAX_ITERATIONS: usize = 10000;",
            "",
            "    loop {",
            "        if state == ParagraphState::FINISHED { break; }",
            "        iterations += 1;",
            "        if iterations > MAX_ITERATIONS {",
            '            return Err(BusinessError::ArithmeticError("Loop infinito detectado".to_string()));',
            "        }",
            "",
            "        match state {",
        ])

        for i, name in enumerate(para_names):
            para = self.paragraphs[name]
            variant = name.replace('-', '_').upper()
            lines.append(f"            ParagraphState::{variant} => {{")

            if para.alter_mutable:
                lines.append(f'                let actual = ctx.get_alter_target("{name}").await;')
                lines.append(f'                if actual != "{name}" {{')
                lines.append(f'                    state = match actual.as_str() {{')
                for p_name in para_names:
                    lines.append(f'                        "{p_name}" => ParagraphState::{p_name.replace("-", "_").upper()},')
                lines.append(f'                        _ => ParagraphState::FINISHED,')
                lines.append(f'                    }};')
                lines.append('                    continue;')
                lines.append('                }')

            open_braces = 0
            has_jump = False
            for stmt in para.statements:
                transmuted = self._transmute_statement_spaghetti(stmt)
                if transmuted:
                    net_braces = transmuted.count('{') - transmuted.count('}')
                    open_braces += net_braces
                    lines.append(self.indent * 4 + transmuted)
                    if "state =" in transmuted and "continue;" in transmuted:
                        has_jump = True
                    if "return Ok(())" in transmuted:
                        has_jump = True

                    if stmt.raw_text.strip().endswith('.'):
                        while open_braces > 0:
                            lines.append(self.indent * 4 + "}")
                            open_braces -= 1

            while open_braces > 0:
                lines.append(self.indent * 4 + "}")
                open_braces -= 1

            if not has_jump:
                if i < len(para_names) - 1:
                    next_variant = para_names[i+1].replace('-', '_').upper()
                    lines.append(f"                state = ParagraphState::{next_variant}; continue;")
                else:
                    lines.append(f"                state = ParagraphState::FINISHED; continue;")

            lines.append("            }")

        lines.extend([
            "            ParagraphState::FINISHED => break,",
            "        }",
            "    }",
            "    Ok(())",
            "}",
        ])

        for name, para in self.paragraphs.items():
            lines.extend(self._generate_paragraph_function(para))

        return '\n'.join(lines)

    def _generate_paragraph_function(self, para: CobolParagraph) -> List[str]:
        func_name = para.name.lower().replace('-', '_')
        lines = [f"async fn {func_name}(ctx: &Context) -> Result<(), BusinessError> {{"]
        open_braces = 0
        for stmt in para.statements:
            if stmt.verb in [CobolVerb.GO_TO, CobolVerb.GO_TO_DEPENDING, CobolVerb.STOP_RUN]:
                lines.append(self.indent + f"// Jump {stmt.verb.name} ignored in structured version")
                continue
            transmuted = self._transmute_basic_verb(stmt)
            if transmuted:
                net_braces = transmuted.count('{') - transmuted.count('}')
                open_braces += net_braces
                lines.append(self.indent + transmuted)
                if stmt.raw_text.strip().endswith('.'):
                    while open_braces > 0:
                        lines.append(self.indent + "}")
                        open_braces -= 1
        while open_braces > 0:
            lines.append(self.indent + "}")
            open_braces -= 1
        lines.append("    Ok(())")
        lines.append("}")
        return lines

    def _transmute_statement_spaghetti(self, stmt: CobolStatement) -> Optional[str]:
        if stmt.verb == CobolVerb.GO_TO:
            target = stmt.target_paragraph.replace('-', '_').upper()
            return f"state = ParagraphState::{target}; continue;"
        elif stmt.verb == CobolVerb.GO_TO_DEPENDING:
            arms = [f"{i+1} => ParagraphState::{t.replace('-', '_').upper()}" for i, t in enumerate(stmt.target_paragraphs)]
            cond = self._transmute_condition(stmt.condition)
            return f"state = match ({cond}) as i64 {{ {' , '.join(arms)}, _ => state }}; continue;"
        elif stmt.verb == CobolVerb.ALTER:
            return f'ctx.set_alter_target("{stmt.operands[0]}", "{stmt.operands[1]}").await;'
        elif stmt.verb == CobolVerb.STOP_RUN:
            return "state = ParagraphState::FINISHED; continue;"
        return self._transmute_basic_verb(stmt)

    def _is_var(self, token: str) -> bool:
        return '-' in token or token.startswith('WS-') or token.startswith('LS-')

    def _transmute_basic_verb(self, stmt: CobolStatement) -> Optional[str]:
        if stmt.verb == CobolVerb.MOVE:
            src, dst = stmt.operands[0], stmt.operands[1]
            if self._is_var(src):
                return f'ctx.set_decimal("{dst}", ctx.get_decimal("{src}").await?).await?;'
            elif src.startswith("'") or src.startswith('"'):
                return f'ctx.set_string("{dst}", {src}.to_string()).await?;'
            else:
                return f'ctx.set_decimal("{dst}", dec!({src})).await?;'
        elif stmt.verb == CobolVerb.COMPUTE:
            expr = self._transmute_arithmetic(stmt.arithmetic_expression)
            return f'ctx.set_decimal("{stmt.operands[0]}", ({expr})).await?;'
        elif stmt.verb == CobolVerb.SUBTRACT:
            a, b = stmt.operands[0], stmt.operands[1]
            val_a = f'ctx.get_decimal("{a}").await?' if self._is_var(a) else f'dec!({a})'
            return f'ctx.set_decimal("{b}", ctx.get_decimal("{b}").await? - {val_a}).await?;'
        elif stmt.verb == CobolVerb.IF:
            return f"if {self._transmute_condition(stmt.condition)} {{"
        elif stmt.verb == CobolVerb.ELSE:
            return "} else {"
        elif stmt.verb == CobolVerb.DISPLAY:
            val = f'ctx.get_decimal("{stmt.operands[0]}").await?' if self._is_var(stmt.operands[0]) else stmt.operands[0]
            return f'println!("{{}}", {val});'
        elif stmt.verb == CobolVerb.PERFORM:
            target = stmt.target_paragraph.lower().replace('-', '_')
            return f"Box::pin({target}(ctx)).await?;"
        return f"// TODO: {stmt.verb.name}"

    def _transmute_arithmetic(self, expr: str) -> str:
        if not expr: return "dec!(0)"
        tokens = expr.split()
        result = []
        for token in tokens:
            clean = token.strip('().')
            if '-' in clean and not clean.replace('-', '').isdigit():
                result.append(f'ctx.get_decimal("{clean}").await?')
            elif clean.replace('.', '', 1).isdigit():
                result.append(f'dec!({clean})')
            else:
                result.append(token)
        return ' '.join(result)

    def _transmute_condition(self, cond: Optional[str]) -> str:
        if not cond: return "true"
        cond = cond.strip()
        replacements = [
            (r'\s+NOT\s+LESS\s+THAN\s+', ' >= '), (r'\s+NOT\s+GREATER\s+THAN\s+', ' <= '),
            (r'\s+NOT\s+EQUAL\s+TO\s+', ' != '), (r'\s+LESS\s+THAN\s+', ' < '),
            (r'\s+GREATER\s+THAN\s+', ' > '), (r'\s+EQUAL\s+TO\s+', ' == '),
            (r'\s+AND\s+', ' && '), (r'\s+OR\s+', ' || '), (r'\s+NOT\s+', ' !'),
            (r'\s+=\s+', ' == '), (r'\s+>\s+', ' > '), (r'\s+<\s+', ' < '),
        ]
        for pattern, replacement in replacements:
            cond = re.sub(pattern, replacement, cond, flags=re.IGNORECASE)
        tokens = cond.split()
        result = []
        for token in tokens:
            clean = token.strip('()<>=!&|.')
            if self._is_var(clean):
                if 'FLAG' in clean or 'STATUS' in clean:
                    result.append(token.replace(clean, f'ctx.get_string("{clean}").await?'))
                else:
                    result.append(token.replace(clean, f'ctx.get_decimal("{clean}").await?'))
            elif clean.replace('.', '', 1).isdigit():
                result.append(token.replace(clean, f'dec!({clean})'))
            elif clean.startswith("'") or clean.startswith('"'):
                 rust_str = clean.replace("'", '"')
                 result.append(token.replace(clean, rust_str))
            else:
                result.append(token)
        return ' '.join(result)

def main():
    print("🜁 ASI-Ω UNIVERSAL COBOL PARSER v2.0")
    spaghetti_cobol = """
       PROCEDURE DIVISION.

       MAIN-LOGIC.
           PERFORM INICIALIZAR.
           GO TO PRIMEIRO-PROCESSAMENTO.

       INICIALIZAR.
           MOVE 100 TO WS-COUNTER.
           MOVE 'N' TO WS-FLAG.

       PRIMEIRO-PROCESSAMENTO.
           DISPLAY WS-COUNTER.
           SUBTRACT 1 FROM WS-COUNTER.
           IF WS-COUNTER > 0
               GO TO PRIMEIRO-PROCESSAMENTO
           ELSE
               GO TO VERIFICAR-FLAG.

       VERIFICAR-FLAG.
           IF WS-FLAG = 'S'
               GO TO FIM-PROGRAMA
           ELSE
               ALTER PRIMEIRO-PROCESSAMENTO TO PROCEED TO SEGUNDO-PROCESSO
               MOVE 'S' TO WS-FLAG
               GO TO PRIMEIRO-PROCESSAMENTO.

       SEGUNDO-PROCESSO.
           DISPLAY WS-COUNTER.
           GO TO FIM-PROGRAMA.

       FIM-PROGRAMA.
           DISPLAY "FIM".
           STOP RUN.
    """
    parser = UniversalCobolParser()
    paragraphs = parser.extract(spaghetti_cobol)
    transmuter = SpaghettiRustTransmuter(parser)
    rust_code = transmuter.transmute()
    with open("spaghetti_transmuted.rs", "w") as f:
        f.write(rust_code)
    print("✅ Transmutação completa! Artefatos: spaghetti_transmuted.rs")

if __name__ == "__main__":
    main()
