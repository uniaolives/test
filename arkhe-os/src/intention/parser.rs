use nom::{
    bytes::complete::{tag, take_while1, take_until},
    character::complete::{char, multispace0, multispace1},
    number::complete::double,
    sequence::{delimited, preceded, tuple},
    IResult,
    multi::many0,
};

#[derive(Debug, PartialEq, Clone)]
pub struct IntentionAst {
    pub name: String,
    pub target: String,
    pub coherence: f64,
    pub priority: i32,
    pub payload: String,
}

fn ws<'a, F, O, E>(inner: F) -> impl FnMut(&'a str) -> IResult<&'a str, O, E>
where
    F: FnMut(&'a str) -> IResult<&'a str, O, E>,
    E: nom::error::ParseError<&'a str>,
{
    delimited(multispace0, inner, multispace0)
}

fn parse_quoted_string(input: &str) -> IResult<&str, &str> {
    delimited(char('"'), take_until("\""), char('"'))(input)
}

#[derive(Debug)]
enum KvPair {
    Target(String),
    Coherence(f64),
    Priority(i32),
    Payload(String),
}

fn parse_kv(input: &str) -> IResult<&str, KvPair> {
    let (input, key) = take_while1(|c: char| c.is_alphanumeric())(input)?;
    let (input, _) = ws(char(':'))(input)?;

    match key {
        "target" => {
            let (input, val) = parse_quoted_string(input)?;
            Ok((input, KvPair::Target(val.to_string())))
        },
        "coherence" => {
            let (input, val) = double(input)?;
            Ok((input, KvPair::Coherence(val)))
        },
        "priority" => {
            let (input, val) = double(input)?; // simple hack for now
            Ok((input, KvPair::Priority(val as i32)))
        },
        "payload" => {
            let (input, val) = parse_quoted_string(input)?;
            Ok((input, KvPair::Payload(val.to_string())))
        },
        _ => Err(nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Tag))),
    }
}

pub fn parse_intention_block(input: &str) -> IResult<&str, IntentionAst> {
    let (input, _) = multispace0(input)?;
    let (input, _) = tag("intention")(input)?;
    let (input, _) = multispace1(input)?;
    let (input, name) = take_while1(|c: char| c.is_alphanumeric() || c == '_')(input)?;
    let (input, _) = ws(char('{'))(input)?;

    let (input, kv_list) = many0(ws(parse_kv))(input)?;
    let (input, _) = ws(char('}'))(input)?;

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
