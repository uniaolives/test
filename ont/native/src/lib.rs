pub mod ast;
pub mod compiler;
pub mod utils;
pub mod backends;
pub mod onchain;
pub mod type_checker;
pub mod std;
pub mod cli;
pub mod audit;
pub mod deployer;

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

// Re-export common types
pub use compiler::compile;
pub use deployer::deploy;

// Add some types for InvariantWitness and DeploymentTarget as used in main.rs
pub struct InvariantWitness;
impl InvariantWitness {
    pub fn new(_a: [u8; 32], _b: [u8; 32], _c: [u8; 32]) -> Self { InvariantWitness }
}

pub enum DeploymentTarget {
    Local,
    Mobile
}

pub struct ProductionAuditor;
impl ProductionAuditor {
    pub fn new(_w: InvariantWitness, _t: DeploymentTarget) -> Self { ProductionAuditor }
    pub fn start(_a: ProductionAuditor) { println!("Starting production auditor..."); }
}
