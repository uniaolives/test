use nom::{
    branch::alt,
    bytes::complete::{is_not, tag, take_while1},
    character::complete::{char, multispace0},
    number::complete::double as nom_f64,
    combinator::{map, recognize, value},
    multi::many0,
    sequence::{delimited, separated_pair},
    IResult,
};

// --- A Árvore Sintática Abstrata (AST) ---
#[derive(Debug, Clone)]
pub struct IntentionAst {
    pub name: String,
    pub target: String,
    pub coherence: f64,
    pub priority: i32,
    pub payload: String,
}

// Enum intermediário para ler os pares chave-valor em qualquer ordem
#[derive(Debug)]
enum KvPair {
    Target(String),
    Coherence(f64),
    Priority(i32),
    Payload(String),
}

// --- Funções Auxiliares de Parsing (Lexer) ---

/// Consome espaços em branco ao redor de um parser
fn ws<'a, F, O>(inner: F) -> impl FnMut(&'a str) -> IResult<&'a str, O>
where
    F: FnMut(&'a str) -> IResult<&'a str, O>,
{
    delimited(multispace0, inner, multispace0)
}

/// Lê uma string entre aspas (ex: "2009-01-03")
fn parse_string_value(input: &str) -> IResult<&str, String> {
    let parser = delimited(char('"'), is_not("\""), char('"'));
    map(parser, |s: &str| s.to_string())(input)
}

/// Mapeia as palavras-chave de prioridade para inteiros do nosso Kernel
fn parse_priority_value(input: &str) -> IResult<&str, i32> {
    alt((
        value(1, tag("low")),
        value(3, tag("normal")),
        value(5, tag("high")),
        value(10, tag("critical")),
    ))(input)
}

// --- Parsers de Chave-Valor ---

fn parse_kv_target(input: &str) -> IResult<&str, KvPair> {
    map(
        separated_pair(tag("target"), ws(char(':')), parse_string_value),
        |(_, val)| KvPair::Target(val),
    )(input)
}

fn parse_kv_coherence(input: &str) -> IResult<&str, KvPair> {
    map(
        separated_pair(tag("coherence"), ws(char(':')), nom_f64),
        |(_, val)| KvPair::Coherence(val),
    )(input)
}

fn parse_kv_priority(input: &str) -> IResult<&str, KvPair> {
    map(
        separated_pair(tag("priority"), ws(char(':')), parse_priority_value),
        |(_, val)| KvPair::Priority(val),
    )(input)
}

fn parse_kv_payload(input: &str) -> IResult<&str, KvPair> {
    map(
        separated_pair(tag("payload"), ws(char(':')), parse_string_value),
        |(_, val)| KvPair::Payload(val),
    )(input)
}

/// Lê qualquer um dos campos válidos dentro do bloco
fn parse_any_kv(input: &str) -> IResult<&str, KvPair> {
    ws(alt((
        parse_kv_target,
        parse_kv_coherence,
        parse_kv_priority,
        parse_kv_payload,
    )))(input)
}

// --- O Parser Principal do Bloco da Intenção ---

/// Lê a estrutura completa: intention <nome> { <campos> }
pub fn parse_intention_block(input: &str) -> IResult<&str, IntentionAst> {
    let (input, _) = ws(tag("intention"))(input)?;

    // Lê o nome da intenção (ex: stabilize_timeline)
    let (input, name) = ws(recognize(take_while1(|c: char| c.is_alphanumeric() || c == '_')))(input)?;

    // Lê os campos dentro das chaves {}
    let (input, kv_list) = delimited(
        ws(char('{')),
        many0(parse_any_kv),
        ws(char('}')),
    )(input)?;

    // Constrói a AST dobrando a lista de atributos
    let mut ast = IntentionAst {
        name: name.to_string(),
        target: String::new(),
        coherence: 0.0,
        priority: 0,
        payload: String::new(),
    };

    for kv in kv_list {
        match kv {
            KvPair::Target(v) => ast.target = v,
            KvPair::Coherence(v) => ast.coherence = v,
            KvPair::Priority(v) => ast.priority = v,
            KvPair::Payload(v) => ast.payload = v,
        }
    }

    Ok((input, ast))
}
