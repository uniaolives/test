# MANIFESTO DA LINGUAGEM SOBERANA (MLS)
## Versão 1.0 - Fase A: Fundação Constitucional (ICMS)
### Data: 2025-01-24
### Status: Especificação para Implementação

---

## 1. VISÃO GERAL CONSTITUCIONAL

A Linguagem Soberana (LS) é uma linguagem de programação de domínio específico (DSL) projetada para expressar lógica de políticas públicas brasileiras com garantias constitucionais verificáveis em tempo de compilação.

### 1.1 Invariantes Constitucionais Mapeados

| Invariante | Característica LS | Mecanismo de Verificação |
|------------|-------------------|-------------------------|
| **I1 (Aterramento Físico)** | Anotações de orçamento energético | Checker estático valida limites térmicos |
| **I2 (Falseabilidade)** | Semântica determinística, logs completos | Transpilação para CCIR verificável |
| **I3 (Soberania)** | Sintaxe em português, sem dependências externas | Compilador nacional, runtime aberto |
| **I4 (Complexidade)** | Garantias de complexidade polinomial | Análise estática de loops/alocações |
| **I5 (Autonomia)** | Primitivas de governança distribuída | Verificação de permissões em tempo de compilação |

---

## 2. GRAMÁTICA FORMAL (EBNF)

```ebnf
(* Programa *)
programa ::= { declaracao } ;

declaracao ::= funcao | processo | tipo | constante ;

(* Funções - cálculos matemáticos verificáveis *)
funcao ::=
    "funcao" identificador "(" parametros ")"
    "->" tipo_retorno
    anotacoes_funcao
    bloco ;

anotacoes_funcao ::=
    { "verificavel_mathematicamente"
    | "com_precisao" "(" "decimais" ":" inteiro ")"
    | "com_complexidade" "(" "O" "(" notacao_o ")" ")" } ;

(* Processos - workflows de governança *)
processo ::=
    "processo" identificador "(" parametros ")"
    "->" tipo_retorno
    anotacoes_processo
    bloco_processo ;

anotacoes_processo ::=
    { "com_rastreamento_completo" "(" invariante ")"
    | "sujeito_a_voto" "(" autoridade "," "quorum" ":" decimal ")"
    | "auditavel_por" "(" orgaos ")" } ;

(* Tipos constitucionais *)
tipo ::=
    "Decimal" | "Inteiro" | "Booleano" | "Texto"
    | "Lista" "<" tipo ">"
    | "Mapeamento" "<" tipo "," tipo ">"
    | identificador ;

(* Blocos e comandos *)
bloco ::= "{" { comando } "}" ;
bloco_processo ::= "{" { etapa } "}" ;

comando ::=
    atribuicao
    | chamada_funcao
    | estrutura_controle
    | retorno ;

etapa ::= "etapa" identificador bloco ;

estrutura_controle ::=
    "para" "cada" identificador "em" expressao bloco
    | "se" expressao "entao" bloco [ "senao" bloco ]
    | "enquanto" expressao bloco ;

(* Expressões *)
expressao ::=
    literal
    | identificador
    | expressao operador expressao
    | chamada_funcao ;

operador ::= "+" | "-" | "*" | "/" | "==" | "!=" | "<" | ">" | "<=" | ">=" ;

literal ::= decimal | inteiro | texto | booleano ;

identificador ::= [a-zA-Z_][a-zA-Z0-9_]* ;
```

---

3. SEMÂNTICA FORMAL

3.1 Regras de Escopo e Rastreabilidade

Toda variável deve ter:
1. Origem explícita (parâmetro, declaração local, ou retorno de função verificável)
2. Fluxo constitucional (não pode ser modificada por agentes não-autorizados)
3. Auditoria automática (cada modificação gera entrada no log constitucional)

3.2 Precisão Numérica

Operações com `Decimal` devem:
- Usar aritmética de ponto fixo (não flutuante) para evitar erros de arredondamento
- Respeitar a precisão especificada em `com_precisao(decimais: n)`
- Propagar erros de precisão em tempo de compilação

3.3 Complexidade Garantida

O checker estático deve verificar:
- Loops `para cada` têm complexidade O(n) onde n é o tamanho da coleção
- Loops `enquanto` devem ter prova de terminação (variante de loop)
- Recursão não é permitida em Fase A (evita stack overflow não-determinístico)

---

4. SISTEMA DE TIPOS COM METADADOS

4.1 Tipos Básicos

```rust
// Representação interna (Rust-like)
enum TipoLS {
    Decimal { precisao: u8 },           // Decimal fixo (ex: 2 casas para moeda)
    Inteiro,                            // i64
    Booleano,                           // bool
    Texto,                              // String UTF-8
    Lista(Box<TipoLS>),                 // Vec<T>
    Mapeamento(Box<TipoLS>, Box<TipoLS>), // HashMap<K,V>
    Estrutura { campos: Vec<Campo> },   // Struct personalizado
}
```

4.2 Metadados Constitucionais

Todo valor carrega:
- Proveniência: Origem do dado (sistema legado, cálculo, entrada usuário)
- Sensibilidade: Nível de proteção (público, restrito, sigiloso)
- Tempo de vida: Quando deve ser descartado/anonimizado

---

5. TRANSPILAÇÃO PARA CCIR

5.1 Dialetos CCIR Utilizados

- ccir-math: Operações matemáticas verificáveis
- ccir-constit: Garantias constitucionais (logs, auditoria)
- ccir-io: Entrada/saída (para Fases B e C)

5.2 Exemplo de Transpilação

Entrada (LS):

```portuguese
funcao calcular_quota(receita: Decimal, populacao: Inteiro) -> Decimal
    verificavel_mathematicamente
    com_precisao(decimais: 2)
{
    retorne receita * populacao / populacao_total;
}
```

Saída (CCIR):

```llvm
; CCIR - Dialeto Math + Constit
define @calcular_quota(%receita: decimal<2>, %populacao: i64) -> decimal<2>
    attributes {
        "constitutional.verifiable" = true,
        "constitutional.precision" = 2,
        "constitutional.source" = "ICMS_Lei_62_89"
    }
{
    %0 = mul_decimal %receita, %populacao
    %1 = div_decimal %0, @populacao_total
    %2 = check_precision %1, 2
    %3 = log_audit %0, %1, "calculo_icms"
    ret %2
}
```

---

6. VALIDAÇÃO E TESTES

6.1 Critérios de Aceitação Fase A

1. Correção Matemática: Resultados idênticos ao SIAFI (±0.01%)
2. Determinismo: Mesma entrada sempre produz mesma saída (mesmo seed)
3. Auditoria: Toda operação gera log verificável
4. Performance: Execução < 100ms para 27 estados (ICMS)

6.2 Casos de Teste

Ver `test_cases/icms/` no repositório.

---

7. ROADMAP

- Fase A (Semanas 1-2): Parser + Transpilador ICMS (este documento)
- Fase B (Semanas 3-4): Workflows de governança (Licitação)
- Fase C (Semanas 5-6): Orçamento termodinâmico (Saúde + I1)

---

8. REFERÊNCIAS

- Constituição Federal, Art. 158 (ICMS)
- Lei Complementar 62/89 (Repartição ICMS)
- Invariantes Constitucionais (I1-I5) - Framework DeAGI
