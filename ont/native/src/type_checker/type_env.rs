use std::collections::{HashMap, HashSet};
use crate::ast::{Type, TypeAnnotation};

pub struct TypeEnv {
    pub functions: HashMap<String, TypeAnnotation>,
    pub side_effects: HashSet<String>,
}

impl TypeEnv {
    pub fn new() -> Self {
        let mut env = Self {
            functions: HashMap::new(),
            side_effects: HashSet::new(),
        };

        // Register the standard library
        crate::std::register_std(&mut env);

        env
    }

    pub fn define_function(&mut self, name: &str, annotation: TypeAnnotation) -> Result<(), String> {
        self.functions.insert(name.to_string(), annotation);
        Ok(())
    }

    pub fn add_side_effect(&mut self, name: &str) {
        self.side_effects.insert(name.to_string());
    }
}
