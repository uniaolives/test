use crate::backends::solidity::CompiledContract;

pub fn extract_constraints(compiled: &CompiledContract) -> Vec<String> {
    // Mock extraction logic
    let mut constraints = Vec::new();
    if compiled.source_code.contains("pureGuard") {
        constraints.push("sandbox".to_string());
    }
    constraints
}
