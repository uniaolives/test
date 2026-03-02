use crate::type_checker::type_env::TypeEnv;
use crate::ast::{Type, TypeAnnotation, Paradigm};

pub fn register(env: &mut TypeEnv) {
    // sha256: Bytes -> Bytes
    // Paradigm: Pure (Matemática pura, sem estado)
    env.define_function("sha256", TypeAnnotation {
        name: "sha256".to_string(),
        type_expr: Type::Function(
            vec![Type::Bytes],
            Box::new(Type::Bytes)
        ),
        paradigm: Paradigm::Functional, // Seguro para usar em blocos Pure
    }).unwrap();

    // ec_recover: Hash, Signature -> Address
    // Paradigm: Pure
    env.define_function("ec_recover", TypeAnnotation {
        name: "ec_recover".to_string(),
        type_expr: Type::Function(
            vec![Type::Bytes, Type::Bytes],
            Box::new(Type::Address)
        ),
        paradigm: Paradigm::Functional,
    }).unwrap();
}
