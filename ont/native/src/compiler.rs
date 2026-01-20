use thiserror::Error;
use crate::ast::Type;

#[derive(Error, Debug)]
pub enum CompilerError {
    #[error("Unsupported type: {0:?}")]
    UnsupportedType(Type),
    #[error("Constraint violation: {0}")]
    ConstraintViolation(String),
    #[error("Other error: {0}")]
    Other(String),
}

pub type CompilerResult<T> = Result<T, CompilerError>;
