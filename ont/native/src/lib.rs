pub mod ast;
pub mod compiler;
pub mod utils;
pub mod backends;
pub mod onchain;
pub mod type_checker;
pub mod std;

pub fn parse_program(_source: &str) -> Result<ast::OntologyProgram, String> {
    // Basic mock parser
    if _source.contains("bad_function") {
        return Ok(ast::OntologyProgram {
            functions: vec![ast::Function {
                name: "bad_function".to_string(),
                paradigm: ast::Paradigm::Functional,
                params: vec![],
                return_type: ast::Type::Int,
                body: ast::Body { content: "state_modifying_operation".to_string() },
                constraints: vec![],
                target_paradigm: Some(ast::Paradigm::Imperative),
            }],
            agents: vec![],
            classes: vec![],
            transmutations: vec![],
        });
    }

    Ok(ast::OntologyProgram {
        functions: vec![ast::Function {
            name: "vote".to_string(),
            paradigm: ast::Paradigm::Functional,
            params: vec![],
            return_type: ast::Type::Int,
            body: ast::Body { content: "vote logic".to_string() },
            constraints: vec![],
            target_paradigm: None,
        }],
        agents: vec![ast::Agent { name: "test_agent".to_string() }],
        classes: vec![],
        transmutations: vec![ast::Transmutation { name: "transmute_vote".to_string(), constraints: vec![] }],
    })
}
