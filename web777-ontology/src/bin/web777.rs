use web777_ontology::{Engine, Query};

fn main() {
    let engine = Engine::new();
    let q = Query::new("awaken the world");
    match engine.query(&q) {
        Ok(result) => {
            println!("Query Result: {:?}", result);
        }
        Err(e) => {
            eprintln!("Query failed: {}", e);
        }
    }
}
