use web777_ontology::{Engine, semantic_query::SemanticQuery};

fn main() {
    let mut engine = Engine::new();
    let q = SemanticQuery::parse("awaken the world").unwrap();
    match engine.query(&q) {
        Ok(result) => {
            println!("Query Result: {:?}", result);
        }
        Err(e) => {
            eprintln!("Query failed: {}", e);
        }
    }
}
