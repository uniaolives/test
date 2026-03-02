use crate::type_checker::type_env::TypeEnv;
use crate::ast::{Type, TypeAnnotation, Paradigm};

pub fn register(env: &mut TypeEnv) {
    // store: key (String), value (Any) -> Unit
    // Paradigm: Imperative (Modifica estado)
    env.define_function("store", TypeAnnotation {
        name: "store".to_string(),
        type_expr: Type::Function(
            vec![Type::String, Type::TypeVar("T".to_string())],
            Box::new(Type::Unit)
        ),
        paradigm: Paradigm::Imperative,
    }).unwrap();

    // load: key (String) -> T
    // Paradigm: Imperative (Lê estado externo - quebra pureza estrita)
    env.define_function("load", TypeAnnotation {
        name: "load".to_string(),
        type_expr: Type::Function(
            vec![Type::String],
            Box::new(Type::TypeVar("T".to_string()))
        ),
        paradigm: Paradigm::Imperative,
    }).unwrap();

    // Marcar como side-effects para o Type Checker
    env.add_side_effect("store");
    env.add_side_effect("load");
}
